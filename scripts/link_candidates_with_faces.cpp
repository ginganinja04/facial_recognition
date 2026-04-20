#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct Detection {
    std::string camera;
    std::string day;
    std::string frame_file;
    std::string timestamp;
    int person_id_in_frame = 0;
    double bbox_area = 0.0;
    int time_minutes = 0;
    std::vector<double> face_embedding;
    bool has_face = false;
};

struct Match {
    std::string detection_id_a;
    std::string detection_id_b;
    std::string camera_a;
    std::string camera_b;
    std::string day_a;
    std::string day_b;
    std::string timestamp_a;
    std::string timestamp_b;
    double face_similarity = 0.0;
    double time_similarity = 0.0;
    double size_similarity = 0.0;
    double combined_score = 0.0;
    int time_diff_minutes = 0;
    double size_diff = 0.0;
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> result;
    std::string current;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            result.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }

    result.push_back(current);
    return result;
}

std::string csv_escape(const std::string& value) {
    bool need_quotes = value.find(',') != std::string::npos || value.find('"') != std::string::npos;
    if (!need_quotes) {
        return value;
    }

    std::ostringstream escaped;
    escaped << '"';
    for (char c : value) {
        if (c == '"') {
            escaped << "\"\"";
        } else {
            escaped << c;
        }
    }
    escaped << '"';
    return escaped.str();
}

bool parse_int(const std::string& text, int& value) {
    try {
        size_t idx;
        value = std::stoi(text, &idx);
        return idx == text.size();
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string& text, double& value) {
    try {
        size_t idx;
        value = std::stod(text, &idx);
        return idx == text.size();
    } catch (...) {
        return false;
    }
}

int parse_time_minutes(const std::string& timestamp) {
    int hour = 0;
    int minute = 0;
    int second = 0;
    char sep1 = 0;
    char sep2 = 0;
    std::istringstream in(timestamp);
    in >> hour >> sep1 >> minute;
    if (in && sep1 == ':') {
        if (in.peek() == ':') {
            in >> sep2 >> second;
        }
    }
    if (!in) {
        return 0;
    }
    return hour * 60 + minute;
}

std::vector<double> parse_embedding(const std::string& text) {
    std::vector<double> embedding;
    if (text.empty()) {
        return embedding;
    }

    std::string trimmed = text;
    if (!trimmed.empty() && trimmed.front() == '"' && trimmed.back() == '"') {
        trimmed = trimmed.substr(1, trimmed.size() - 2);
    }

    std::istringstream in(trimmed);
    std::string token;
    while (std::getline(in, token, ',')) {
        double value;
        if (parse_double(token, value)) {
            embedding.push_back(value);
        } else {
            embedding.clear();
            break;
        }
    }

    return embedding;
}

double bounded_similarity(double value) {
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

double face_similarity(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != 128 || b.size() != 128) {
        return 0.0;
    }

    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < 128; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a <= 0.0 || norm_b <= 0.0) {
        return 0.0;
    }

    double cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    return bounded_similarity(cosine);
}

double size_similarity(double area_a, double area_b) {
    if (area_a <= 0.0 || area_b <= 0.0) {
        return 0.0;
    }

    double diff = std::fabs(area_a - area_b);
    double maximum = std::max(area_a, area_b);
    return bounded_similarity(1.0 - diff / maximum);
}

std::vector<Detection> load_detections(const std::string& detections_dir) {
    std::vector<Detection> detections;

    for (const auto& entry : fs::recursive_directory_iterator(detections_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const auto& path = entry.path();
        if (path.extension() != ".csv") {
            continue;
        }

        std::ifstream in(path);
        if (!in) {
            std::cerr << "[WARN] Could not open CSV: " << path << "\n";
            continue;
        }

        std::string header_line;
        if (!std::getline(in, header_line)) {
            continue;
        }

        auto headers = split_csv_line(header_line);
        std::unordered_map<std::string, int> header_index;
        for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
            header_index[headers[i]] = i;
        }

        std::string row;
        while (std::getline(in, row)) {
            if (row.empty()) {
                continue;
            }

            auto fields = split_csv_line(row);
            if (fields.size() < headers.size()) {
                continue;
            }

            Detection det;
            det.camera = fields[header_index["camera"]];
            det.day = fields[header_index["day"]];
            det.frame_file = fields[header_index["frame_file"]];
            det.timestamp = fields[header_index["timestamp"]];
            det.time_minutes = parse_time_minutes(det.timestamp);

            if (header_index.count("person_id_in_frame")) {
                parse_int(fields[header_index["person_id_in_frame"]], det.person_id_in_frame);
            }

            if (header_index.count("bbox_area")) {
                parse_double(fields[header_index["bbox_area"]], det.bbox_area);
            }

            if (header_index.count("face_embedding")) {
                det.face_embedding = parse_embedding(fields[header_index["face_embedding"]]);
                det.has_face = !det.face_embedding.empty();
            }

            detections.push_back(std::move(det));
        }
    }

    return detections;
}

Match compute_similarity(const Detection& a, const Detection& b) {
    Match match;
    match.camera_a = a.camera;
    match.camera_b = b.camera;
    match.day_a = a.day;
    match.day_b = b.day;
    match.timestamp_a = a.timestamp;
    match.timestamp_b = b.timestamp;
    match.detection_id_a = a.camera + "_" + a.day + "_" + a.frame_file + "_" + std::to_string(a.person_id_in_frame);
    match.detection_id_b = b.camera + "_" + b.day + "_" + b.frame_file + "_" + std::to_string(b.person_id_in_frame);
    match.time_diff_minutes = std::abs(a.time_minutes - b.time_minutes);
    match.size_diff = std::fabs(a.bbox_area - b.bbox_area);

    match.time_similarity = bounded_similarity(1.0 - static_cast<double>(match.time_diff_minutes) / 60.0);
    match.size_similarity = size_similarity(a.bbox_area, b.bbox_area);
    match.face_similarity = face_similarity(a.face_embedding, b.face_embedding);

    const double face_weight = 0.4;
    const double time_weight = 0.3;
    const double size_weight = 0.3;
    match.combined_score = face_weight * match.face_similarity +
                           time_weight * match.time_similarity +
                           size_weight * match.size_similarity;

    return match;
}

std::vector<Match> find_candidate_matches(
    const std::vector<Detection>& detections,
    int max_time_diff,
    double min_combined_score,
    int max_candidates_per_detection
) {
    std::unordered_map<std::string, std::vector<Detection>> groups;
    for (const auto& det : detections) {
        std::string key = det.camera + "|" + det.day;
        groups[key].push_back(det);
    }

    std::vector<Match> matches;
    const auto start = std::chrono::high_resolution_clock::now();
    size_t total_detections = detections.size();
    size_t processed = 0;
    size_t progress_step = std::max<size_t>(1, total_detections / 50);

    // Within-camera matching
    std::cout << "[INFO] Performing within-camera matching...\n";
    for (auto& kv : groups) {
        auto& group = kv.second;
        std::sort(group.begin(), group.end(), [](const Detection& a, const Detection& b) {
            return a.time_minutes < b.time_minutes;
        });

        for (size_t i = 0; i < group.size(); ++i) {
            std::vector<Match> candidates;
            const auto& current = group[i];

            for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
                int time_diff = current.time_minutes - group[j].time_minutes;
                if (time_diff > max_time_diff) {
                    break;
                }
                Match candidate = compute_similarity(current, group[j]);
                bool is_consecutive = candidate.time_diff_minutes <= 1 && candidate.size_similarity > 0.7;
                if (candidate.combined_score >= min_combined_score || is_consecutive) {
                    candidates.push_back(std::move(candidate));
                }
            }

            for (size_t j = i + 1; j < group.size(); ++j) {
                int time_diff = group[j].time_minutes - current.time_minutes;
                if (time_diff > max_time_diff) {
                    break;
                }
                Match candidate = compute_similarity(current, group[j]);
                bool is_consecutive = candidate.time_diff_minutes <= 1 && candidate.size_similarity > 0.7;
                if (candidate.combined_score >= min_combined_score || is_consecutive) {
                    candidates.push_back(std::move(candidate));
                }
            }

            std::sort(candidates.begin(), candidates.end(), [](const Match& a, const Match& b) {
                return a.combined_score > b.combined_score;
            });

            int keep = std::min(static_cast<int>(candidates.size()), max_candidates_per_detection);
            for (int k = 0; k < keep; ++k) {
                matches.push_back(std::move(candidates[k]));
            }

            ++processed;
            if (processed % progress_step == 0 || processed == total_detections) {
                double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
                double progress = static_cast<double>(processed) / static_cast<double>(total_detections);
                double remaining = progress > 0.0 ? elapsed * (1.0 - progress) / progress : 0.0;
                std::cout << "\r[PROGRESS] " << std::fixed << std::setprecision(1)
                          << (progress * 100.0) << "% (" << processed << "/" << total_detections << ") "
                          << "elapsed " << std::setprecision(2) << elapsed << "s, "
                          << "remaining " << remaining << "s   " << std::flush;
            }
        }
    }

    // Cross-camera matching (optimized by time windows)
    std::cout << "\n[INFO] Performing cross-camera matching...\n";
    const int TIME_WINDOW_MINUTES = 5;  // Compare detections within 5-minute windows

    // Group all detections by time window
    std::unordered_map<int, std::vector<std::pair<std::string, const Detection*>>> time_windows;
    for (const auto& kv : groups) {
        const auto& group = kv.second;
        for (const auto& det : group) {
            if (!det.has_face) continue;
            int window = det.time_minutes / TIME_WINDOW_MINUTES;
            time_windows[window].emplace_back(kv.first, &det);
        }
    }

    std::cout << "[INFO] Created " << time_windows.size() << " time windows for cross-camera matching\n";

    size_t cross_matches_found = 0;
    for (const auto& kv : time_windows) {
        const auto& window_detections = kv.second;

        // Group by camera within this time window
        std::unordered_map<std::string, std::vector<const Detection*>> cameras_in_window;
        for (const auto& item : window_detections) {
            std::string camera = item.first.substr(0, item.first.find('|'));
            cameras_in_window[camera].push_back(item.second);
        }

        // Compare between different cameras in this time window
        std::vector<std::string> camera_keys;
        for (const auto& cam_kv : cameras_in_window) {
            camera_keys.push_back(cam_kv.first);
        }

        for (size_t i = 0; i < camera_keys.size(); ++i) {
            for (size_t j = i + 1; j < camera_keys.size(); ++j) {
                const auto& cam_a_dets = cameras_in_window[camera_keys[i]];
                const auto& cam_b_dets = cameras_in_window[camera_keys[j]];

                // Compare all detections between these two cameras in this time window
                for (const auto* det_a : cam_a_dets) {
                    for (const auto* det_b : cam_b_dets) {
                        Match candidate = compute_similarity(*det_a, *det_b);
                        if (candidate.combined_score >= min_combined_score) {
                            matches.push_back(std::move(candidate));
                            ++cross_matches_found;
                        }
                    }
                }
            }
        }
    }

    std::cout << "[INFO] Found " << cross_matches_found << " cross-camera matches\n";

    if (!detections.empty()) {
        std::cout << "\r[PROGRESS] 100.0% (" << total_detections << "/" << total_detections << ") completed.\n";
    }

    return matches;
}

void write_matches(const std::string& output_path, const std::vector<Match>& matches) {
    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "[ERROR] Could not open output file: " << output_path << "\n";
        return;
    }

    out << "detection_id_a,detection_id_b,camera_a,camera_b,day_a,day_b,timestamp_a,timestamp_b,"
           "face_similarity,time_similarity,size_similarity,combined_score,time_diff_minutes,size_diff\n";

    out << std::fixed << std::setprecision(6);
    for (const auto& match : matches) {
        out << csv_escape(match.detection_id_a) << ','
            << csv_escape(match.detection_id_b) << ','
            << csv_escape(match.camera_a) << ','
            << csv_escape(match.camera_b) << ','
            << csv_escape(match.day_a) << ','
            << csv_escape(match.day_b) << ','
            << csv_escape(match.timestamp_a) << ','
            << csv_escape(match.timestamp_b) << ','
            << match.face_similarity << ','
            << match.time_similarity << ','
            << match.size_similarity << ','
            << match.combined_score << ','
            << match.time_diff_minutes << ','
            << match.size_diff << '\n';
    }
}

void print_statistics(const std::vector<Match>& matches) {
    if (matches.empty()) {
        std::cout << "[INFO] No candidate matches found.\n";
        return;
    }

    double sum_combined = 0.0;
    double sum_face = 0.0;
    double min_combined = matches.front().combined_score;
    double max_combined = matches.front().combined_score;
    double min_face = matches.front().face_similarity;
    double max_face = matches.front().face_similarity;

    for (const auto& m : matches) {
        sum_combined += m.combined_score;
        sum_face += m.face_similarity;
        min_combined = std::min(min_combined, m.combined_score);
        max_combined = std::max(max_combined, m.combined_score);
        min_face = std::min(min_face, m.face_similarity);
        max_face = std::max(max_face, m.face_similarity);
    }

    double avg_combined = sum_combined / matches.size();
    double avg_face = sum_face / matches.size();

    std::cout << "[STATS] Candidate matches: " << matches.size() << "\n";
    std::cout << "[STATS] Combined score min=" << min_combined
              << " avg=" << avg_combined
              << " max=" << max_combined << "\n";
    std::cout << "[STATS] Face similarity min=" << min_face
              << " avg=" << avg_face
              << " max=" << max_face << "\n";
}

int main(int argc, char* argv[]) {
    std::string detections_dir = "data/detections";
    std::string output_file = "data/summaries/candidate_links_with_faces.csv";
    int max_time_diff = 30;
    double min_score = 0.6;
    int max_candidates = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--detections-dir" && i + 1 < argc) {
            detections_dir = argv[++i];
        } else if (arg == "--output-file" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--max-time-diff" && i + 1 < argc) {
            max_time_diff = std::stoi(argv[++i]);
        } else if (arg == "--min-score" && i + 1 < argc) {
            min_score = std::stod(argv[++i]);
        } else if (arg == "--max-candidates" && i + 1 < argc) {
            max_candidates = std::stoi(argv[++i]);
        }
    }

    const auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "[INFO] Loading detections from: " << detections_dir << "\n";
    auto detections = load_detections(detections_dir);
    const auto after_load = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Loaded " << detections.size() << " detections\n";

    if (detections.empty()) {
        std::cerr << "[ERROR] No detections found.\n";
        return 1;
    }

    std::cout << "[INFO] Finding candidate matches...\n";
    auto matches = find_candidate_matches(detections, max_time_diff, min_score, max_candidates);
    const auto after_match = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Found " << matches.size() << " candidate matches\n";

    double load_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(after_load - start_time).count();
    double match_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(after_match - after_load).count();
    double write_estimate = std::max(0.05, matches.size() * 0.0001);
    std::cout << "[INFO] Elapsed so far: " << std::fixed << std::setprecision(2) << (load_seconds + match_seconds) << " sec. "
              << "Estimated remaining write time: " << write_estimate << " sec.\n";

    fs::create_directories(fs::path(output_file).parent_path());
    write_matches(output_file, matches);
    std::cout << "[OK] Saved matches to " << output_file << "\n";
    print_statistics(matches);

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[INFO] Total runtime: " << (elapsed.count() / 1000.0) << " seconds (" << elapsed.count() << " ms)\n";

    return 0;
}
