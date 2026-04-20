#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

struct Detection {
    std::string camera;
    std::string day;
    std::string frame_file;
    std::string timestamp;
    std::string frame_path;
    std::string detection_id;
    int person_id_in_frame = 0;
    double confidence = 0.0;
    double bbox_area = 0.0;
    int time_minutes = 0;
    bool has_face = false;
};

struct Edge {
    std::string target;
    double score = 0.0;
    std::string camera;
    int time_minutes = 0;
    bool camera_change = false;
};

struct PathInfo {
    std::vector<std::string> path;
    double average_score = 0.0;
    int length = 0;
    int camera_changes = 0;
    double ranking_score = 0.0;
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
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
            tokens.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }

    tokens.push_back(current);
    return tokens;
}

std::string csv_escape(const std::string& value) {
    bool need_quotes = value.find(',') != std::string::npos || value.find('"') != std::string::npos;
    if (!need_quotes) {
        return value;
    }

    std::ostringstream out;
    out << '"';
    for (char c : value) {
        if (c == '"') {
            out << "\"\"";
        } else {
            out << c;
        }
    }
    out << '"';
    return out.str();
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
    return hour * 3600 + minute * 60 + second;
}

bool parse_bool(const std::string& text) {
    std::string value = text;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value == "true" || value == "1" || value == "yes";
}

std::string build_detection_id(const std::string& camera, const std::string& day, const std::string& frame_file, int person_id) {
    return camera + "_" + day + "_" + frame_file + "_" + std::to_string(person_id);
}

bool is_outside_camera(const std::string& camera) {
    return camera == "balcony" || camera == "street_view";
}

bool is_inside_camera(const std::string& camera) {
    return camera == "bar_stage" || camera == "inside_bar";
}

bool can_coexist_at_time(const std::string& camera_a, const std::string& camera_b) {
    // Both outside: OK (street_view and balcony could capture same person at same time)
    if (is_outside_camera(camera_a) && is_outside_camera(camera_b)) {
        return true;
    }
    // Both inside: OK (bar_stage and inside_bar could capture same person at same time)
    if (is_inside_camera(camera_a) && is_inside_camera(camera_b)) {
        return true;
    }
    // One outside and one inside: NOT OK (physically impossible)
    return false;
}

std::unordered_map<std::string, Detection> load_detections(const std::string& detections_dir) {
    std::unordered_map<std::string, Detection> detections;
    size_t loaded_files = 0;

    for (const auto& entry : fs::recursive_directory_iterator(detections_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() != ".csv") {
            continue;
        }

        std::ifstream in(entry.path());
        if (!in) {
            std::cerr << "[WARN] Could not open detection CSV: " << entry.path() << "\n";
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
            if (static_cast<int>(fields.size()) < static_cast<int>(headers.size())) {
                continue;
            }

            Detection det;
            det.camera = fields[header_index["camera"]];
            det.day = fields[header_index["day"]];
            det.frame_file = fields[header_index["frame_file"]];
            det.timestamp = fields[header_index["timestamp"]];
            det.frame_path = header_index.count("frame_path") ? fields[header_index["frame_path"]] : "";
            det.time_minutes = parse_time_minutes(det.timestamp);

            if (header_index.count("person_id_in_frame")) {
                parse_int(fields[header_index["person_id_in_frame"]], det.person_id_in_frame);
            }
            if (header_index.count("confidence")) {
                parse_double(fields[header_index["confidence"]], det.confidence);
            }
            if (header_index.count("bbox_area")) {
                parse_double(fields[header_index["bbox_area"]], det.bbox_area);
            }
            if (header_index.count("has_face")) {
                det.has_face = parse_bool(fields[header_index["has_face"]]);
            }

            det.detection_id = build_detection_id(det.camera, det.day, det.frame_file, det.person_id_in_frame);
            detections.emplace(det.detection_id, std::move(det));
        }

        ++loaded_files;
    }

    std::cout << "[INFO] Loaded " << detections.size() << " detections from " << loaded_files << " CSV files\n";
    return detections;
}

bool parse_detection_id(const std::string& detection_id, std::string& camera, std::string& day, std::string& frame_file, int& person_id) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : detection_id) {
        if (c == '_') {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    parts.push_back(current);

    if (parts.size() < 4) {
        return false;
    }

    camera = parts[0];
    day = parts[1];
    person_id = std::stoi(parts.back());
    frame_file.clear();
    for (size_t i = 2; i + 1 < parts.size(); ++i) {
        if (i > 2) {
            frame_file.push_back('_');
        }
        frame_file += parts[i];
    }
    return true;
}

std::unordered_map<std::string, std::vector<Edge>> build_graph(
    const std::string& matches_csv,
    const std::unordered_map<std::string, Detection>& detection_lookup,
    double min_score,
    double min_confidence,
    bool require_face,
    size_t max_neighbors
) {
    std::unordered_map<std::string, std::vector<Edge>> graph;
    std::ifstream in(matches_csv);
    if (!in) {
        std::cerr << "[ERROR] Could not open matches CSV: " << matches_csv << "\n";
        return graph;
    }

    std::string header_line;
    if (!std::getline(in, header_line)) {
        return graph;
    }

    auto headers = split_csv_line(header_line);
    std::unordered_map<std::string, int> header_index;
    for (int i = 0; i < static_cast<int>(headers.size()); ++i) {
        header_index[headers[i]] = i;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }

    size_t total_matches = lines.size();
    size_t processed = 0;
    size_t progress_step = std::max<size_t>(1, total_matches / 50);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& row : lines) {
        auto fields = split_csv_line(row);
        if (fields.size() < headers.size()) {
            ++processed;
            continue;
        }

        double combined_score = 0.0;
        if (header_index.count("combined_score")) {
            parse_double(fields[header_index["combined_score"]], combined_score);
        }
        if (combined_score < min_score) {
            ++processed;
            if (processed % progress_step == 0 || processed == total_matches) {
                auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time).count();
                double progress = static_cast<double>(processed) / static_cast<double>(total_matches);
                double remaining = progress > 0.0 ? elapsed * (1.0 - progress) / progress : 0.0;
                std::cout << "\r[GRAPH] " << std::fixed << std::setprecision(1) << (progress * 100.0)
                          << "% (" << processed << "/" << total_matches << ") elapsed " << std::setprecision(1)
                          << elapsed << "s, remaining " << remaining << "s" << std::flush;
            }
            continue;
        }

        std::string id_a = fields[header_index["detection_id_a"]];
        std::string id_b = fields[header_index["detection_id_b"]];
        std::string camera_a = header_index.count("camera_a") ? fields[header_index["camera_a"]] : "";
        std::string camera_b = header_index.count("camera_b") ? fields[header_index["camera_b"]] : "";

        auto it_a = detection_lookup.find(id_a);
        auto it_b = detection_lookup.find(id_b);
        if (it_a == detection_lookup.end() || it_b == detection_lookup.end()) {
            ++processed;
            if (processed % progress_step == 0 || processed == total_matches) {
                auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time).count();
                double progress = static_cast<double>(processed) / static_cast<double>(total_matches);
                double remaining = progress > 0.0 ? elapsed * (1.0 - progress) / progress : 0.0;
                std::cout << "\r[GRAPH] " << std::fixed << std::setprecision(1) << (progress * 100.0)
                          << "% (" << processed << "/" << total_matches << ") elapsed " << std::setprecision(1)
                          << elapsed << "s, remaining " << remaining << "s" << std::flush;
            }
            continue;
        }

        const Detection& det_a = it_a->second;
        const Detection& det_b = it_b->second;

        if (det_a.confidence < min_confidence || det_b.confidence < min_confidence) {
            ++processed;
            if (processed % progress_step == 0 || processed == total_matches) {
                auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time).count();
                double progress = static_cast<double>(processed) / static_cast<double>(total_matches);
                double remaining = progress > 0.0 ? elapsed * (1.0 - progress) / progress : 0.0;
                std::cout << "\r[GRAPH] " << std::fixed << std::setprecision(1) << (progress * 100.0)
                          << "% (" << processed << "/" << total_matches << ") elapsed " << std::setprecision(1)
                          << elapsed << "s, remaining " << remaining << "s" << std::flush;
            }
            continue;
        }

        if (require_face && (!det_a.has_face || !det_b.has_face)) {
            ++processed;
            if (processed % progress_step == 0 || processed == total_matches) {
                auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time).count();
                double progress = static_cast<double>(processed) / static_cast<double>(total_matches);
                double remaining = progress > 0.0 ? elapsed * (1.0 - progress) / progress : 0.0;
                std::cout << "\r[GRAPH] " << std::fixed << std::setprecision(1) << (progress * 100.0)
                          << "% (" << processed << "/" << total_matches << ") elapsed " << std::setprecision(1)
                          << elapsed << "s, remaining " << remaining << "s" << std::flush;
            }
            continue;
        }

        if (id_a == id_b ||
            (det_a.camera == det_b.camera && det_a.day == det_b.day && det_a.frame_file == det_b.frame_file)) {
            ++processed;
            if (processed % progress_step == 0 || processed == total_matches) {
                auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time).count();
                double progress = static_cast<double>(processed) / static_cast<double>(total_matches);
                double remaining = progress > 0.0 ? elapsed * (1.0 - progress) / progress : 0.0;
                std::cout << "\r[GRAPH] " << std::fixed << std::setprecision(1) << (progress * 100.0)
                          << "% (" << processed << "/" << total_matches << ") elapsed " << std::setprecision(1)
                          << elapsed << "s, remaining " << remaining << "s" << std::flush;
            }
            continue;
        }

        const Detection* source = &det_a;
        const Detection* target = &det_b;
        bool camera_change = (det_a.camera != det_b.camera);
        if (det_b.time_minutes < det_a.time_minutes ||
            (det_b.time_minutes == det_a.time_minutes && id_b < id_a)) {
            source = &det_b;
            target = &det_a;
        }

        // NEW: Reject edges between incompatible simultaneous locations
        // If source and target are at the same time, their cameras must be compatible
        if (source->time_minutes == target->time_minutes) {
            if (!can_coexist_at_time(source->camera, target->camera)) {
                ++processed;
                if (processed % progress_step == 0 || processed == total_matches) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time).count();
                    double progress = static_cast<double>(processed) / static_cast<double>(total_matches);
                    double remaining = progress > 0.0 ? elapsed * (1.0 - progress) / progress : 0.0;
                    std::cout << "\r[GRAPH] " << std::fixed << std::setprecision(1) << (progress * 100.0)
                              << "% (" << processed << "/" << total_matches << ") elapsed " << std::setprecision(1)
                              << elapsed << "s, remaining " << remaining << "s" << std::flush;
                }
                continue;
            }
        }

        Edge directed_edge;
        directed_edge.target = target->detection_id;
        directed_edge.score = combined_score;
        directed_edge.camera = target->camera;
        directed_edge.time_minutes = target->time_minutes;
        directed_edge.camera_change = camera_change;
        graph[source->detection_id].push_back(std::move(directed_edge));

        ++processed;
        if (processed % progress_step == 0 || processed == total_matches) {
            auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time).count();
            double progress = static_cast<double>(processed) / static_cast<double>(total_matches);
            double remaining = progress > 0.0 ? elapsed * (1.0 - progress) / progress : 0.0;
            std::cout << "\r[GRAPH] " << std::fixed << std::setprecision(1) << (progress * 100.0)
                      << "% (" << processed << "/" << total_matches << ") elapsed " << std::setprecision(1)
                      << elapsed << "s, remaining " << remaining << "s" << std::flush;
            if (processed == total_matches) {
                std::cout << '\n';
            }
        }
    }

    for (auto& kv : graph) {
        auto& edges = kv.second;
        std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.score > b.score;
        });
        if (edges.size() > max_neighbors) {
            edges.resize(max_neighbors);
        }
    }

    std::cout << "\n[INFO] Graph built with " << graph.size() << " nodes" << std::endl;
    return graph;
}

double score_path(int length, int camera_changes, double average_score) {
    double length_score = static_cast<double>(length) / 6.0;
    double camera_score = std::min(static_cast<double>(camera_changes) / 3.0, 1.0);
    return 0.4 * length_score + 0.3 * camera_score + 0.3 * average_score;
}

struct PathMinHeap {
    bool operator()(const PathInfo& a, const PathInfo& b) const {
        return a.ranking_score > b.ranking_score;
    }
};

std::vector<PathInfo> find_person_paths(
    const std::unordered_map<std::string, std::vector<Edge>>& graph,
    const std::unordered_map<std::string, Detection>& detection_lookup,
    int max_length,
    int min_camera_changes,
    int max_start_nodes,
    int max_paths
) {
    std::vector<PathInfo> paths;
    std::vector<std::pair<std::string, double>> scored_starts;
    scored_starts.reserve(graph.size());
    for (const auto& kv : graph) {
        if (kv.second.empty()) {
            continue;
        }
        double best_score = 0.0;
        for (const auto& edge : kv.second) {
            best_score = std::max(best_score, edge.score);
        }
        scored_starts.emplace_back(kv.first, best_score);
    }

    if (scored_starts.empty()) {
        return paths;
    }

    std::sort(scored_starts.begin(), scored_starts.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    if (max_start_nodes > 0 && static_cast<size_t>(max_start_nodes) < scored_starts.size()) {
        scored_starts.resize(static_cast<size_t>(max_start_nodes));
    }

    size_t total_nodes = scored_starts.size();
    size_t processed = 0;
    size_t progress_step = std::max<size_t>(1, total_nodes / 50);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::string> current_path;
    current_path.reserve(max_length);
    std::priority_queue<PathInfo, std::vector<PathInfo>, PathMinHeap> top_paths;
    size_t max_candidates = static_cast<size_t>(std::max(1, max_paths) * 10);

    std::function<void(const std::string&, double, int, int)> dfs;
    dfs = [&](const std::string& node, double score_sum, int camera_changes, int previous_time) {
        current_path.push_back(node);

        if (static_cast<int>(current_path.size()) >= 2 && camera_changes >= min_camera_changes) {
            PathInfo info;
            info.path = current_path;
            info.length = static_cast<int>(current_path.size());
            info.average_score = score_sum / info.length;
            info.camera_changes = camera_changes;
            info.ranking_score = score_path(info.length, info.camera_changes, info.average_score);
            if (top_paths.size() < max_candidates) {
                top_paths.push(info);
            } else if (info.ranking_score > top_paths.top().ranking_score) {
                top_paths.pop();
                top_paths.push(info);
            }
        }

        if (static_cast<int>(current_path.size()) < max_length) {
            auto it = graph.find(node);
            if (it != graph.end()) {
                for (const auto& edge : it->second) {
                    if (edge.time_minutes < previous_time) {
                        continue;
                    }
                    if (std::find(current_path.begin(), current_path.end(), edge.target) != current_path.end()) {
                        continue;
                    }
                    
                    // NEW: Check for impossible simultaneous locations
                    // If the edge target is at the same time as current node, verify cameras are compatible
                    if (edge.time_minutes == previous_time) {
                        auto it_current = detection_lookup.find(node);
                        auto it_target = detection_lookup.find(edge.target);
                        if (it_current != detection_lookup.end() && it_target != detection_lookup.end()) {
                            const std::string& current_camera = it_current->second.camera;
                            const std::string& target_camera = it_target->second.camera;
                            
                            // Reject if cameras are from incompatible location types
                            if (!can_coexist_at_time(current_camera, target_camera)) {
                                continue;
                            }
                        }
                    }
                    
                    int next_camera_changes = camera_changes + (edge.camera_change ? 1 : 0);
                    dfs(edge.target, score_sum + edge.score, next_camera_changes, edge.time_minutes);
                }
            }
        }

        current_path.pop_back();
    };

    for (const auto& start_node : scored_starts) {
        auto it = detection_lookup.find(start_node.first);
        if (it == detection_lookup.end()) {
            ++processed;
            continue;
        }
        dfs(start_node.first, 0.0, 0, it->second.time_minutes);
        ++processed;
        if (processed % progress_step == 0 || processed == total_nodes) {
            auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time).count();
            double progress = static_cast<double>(processed) / static_cast<double>(total_nodes);
            double remaining = progress > 0.0 ? elapsed * (1.0 - progress) / progress : 0.0;
            std::cout << "\r[PROGRESS] " << std::fixed << std::setprecision(1) << (progress * 100.0)
                      << "% (" << processed << "/" << total_nodes << ") elapsed " << std::setprecision(1)
                      << elapsed << "s, remaining " << remaining << "s" << std::flush;
            if (processed == total_nodes) {
                std::cout << '\n';
            }
        }
    }

    paths.reserve(top_paths.size());
    while (!top_paths.empty()) {
        paths.push_back(std::move(top_paths.top()));
        top_paths.pop();
    }
    std::sort(paths.begin(), paths.end(), [](const PathInfo& a, const PathInfo& b) {
        return a.ranking_score > b.ranking_score;
    });

    std::unordered_set<std::string> seen_paths;
    std::vector<PathInfo> unique_paths;
    for (const auto& path_info : paths) {
        std::string path_key;
        for (size_t i = 0; i < path_info.path.size(); ++i) {
            if (i > 0) path_key += "|";
            path_key += path_info.path[i];
        }
        if (seen_paths.find(path_key) == seen_paths.end()) {
            seen_paths.insert(path_key);
            unique_paths.push_back(path_info);
        }
    }
    
    std::cout << "[INFO] Found " << paths.size() << " raw paths (" << unique_paths.size() << " unique)" << std::endl;
    return unique_paths;
}

void write_paths(const std::string& output_file, const std::vector<PathInfo>& paths, const std::unordered_map<std::string, Detection>& detection_lookup) {
    std::ofstream out(output_file);
    if (!out) {
        std::cerr << "[ERROR] Could not open output file: " << output_file << "\n";
        return;
    }

    out << "path_id,path_length,camera_changes,average_score,total_score,cameras,days,timestamps,detection_ids,frame_files\n";
    for (size_t i = 0; i < paths.size(); ++i) {
        const auto& path_info = paths[i];
        std::string cameras;
        std::string days;
        std::string timestamps;
        std::string detection_ids;
        std::string frame_files;

        for (size_t j = 0; j < path_info.path.size(); ++j) {
            const std::string& det_id = path_info.path[j];
            const auto it = detection_lookup.find(det_id);
            std::string camera = "";
            std::string day = "";
            std::string timestamp = "";
            std::string frame_file = "";
            if (it != detection_lookup.end()) {
                camera = it->second.camera;
                day = it->second.day;
                timestamp = it->second.timestamp;
                frame_file = it->second.frame_file;
            }
            if (j > 0) {
                cameras += "|";
                days += "|";
                timestamps += "|";
                detection_ids += "|";
                frame_files += "|";
            }
            cameras += camera;
            days += day;
            timestamps += timestamp;
            detection_ids += det_id;
            frame_files += frame_file;
        }

        out << i << ','
            << path_info.length << ','
            << path_info.camera_changes << ','
            << std::fixed << std::setprecision(6) << path_info.average_score << ','
            << std::fixed << std::setprecision(6) << path_info.ranking_score << ','
            << csv_escape(cameras) << ','
            << csv_escape(days) << ','
            << csv_escape(timestamps) << ','
            << csv_escape(detection_ids) << ','
            << csv_escape(frame_files) << '\n';
    }

    std::cout << "[INFO] Saved " << paths.size() << " paths to " << output_file << "\n";
}

int main(int argc, char* argv[]) {
    std::string matches_csv = "data/summaries/candidate_links_with_faces.csv";
    std::string detections_dir = "data/detections";
    std::string output_file = "data/summaries/person_paths.csv";
    int max_path_length = 6;
    int min_camera_changes = 0;
    double min_score = 0.5;
    double min_confidence = 0.45;
    bool require_face = false;
    int max_paths = 100;
    int max_start_nodes = 20000;
    int max_neighbors = 8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--matches-csv" && i + 1 < argc) {
            matches_csv = argv[++i];
        } else if (arg == "--detections-dir" && i + 1 < argc) {
            detections_dir = argv[++i];
        } else if (arg == "--output-file" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--max-path-length" && i + 1 < argc) {
            max_path_length = std::stoi(argv[++i]);
        } else if (arg == "--min-camera-changes" && i + 1 < argc) {
            min_camera_changes = std::stoi(argv[++i]);
        } else if (arg == "--min-score" && i + 1 < argc) {
            min_score = std::stod(argv[++i]);
        } else if (arg == "--min-confidence" && i + 1 < argc) {
            min_confidence = std::stod(argv[++i]);
        } else if (arg == "--require-face") {
            require_face = true;
        } else if (arg == "--max-paths" && i + 1 < argc) {
            max_paths = std::stoi(argv[++i]);
        } else if (arg == "--max-start-nodes" && i + 1 < argc) {
            max_start_nodes = std::stoi(argv[++i]);
        } else if (arg == "--max-neighbors" && i + 1 < argc) {
            max_neighbors = std::stoi(argv[++i]);
        }
    }

    std::cout << "[INFO] Loading detections...\n";
    auto detection_lookup = load_detections(detections_dir);
    if (detection_lookup.empty()) {
        std::cerr << "[ERROR] No detections loaded.\n";
        return 1;
    }

    std::cout << "[INFO] Building match graph...\n";
    auto graph = build_graph(matches_csv, detection_lookup, min_score, min_confidence, require_face, static_cast<size_t>(max_neighbors));
    std::cout << "[INFO] Graph has " << graph.size() << " nodes\n";

    std::cout << "[INFO] Finding person paths...\n";
    auto raw_paths = find_person_paths(graph, detection_lookup, max_path_length, min_camera_changes, max_start_nodes, max_paths);
    std::cout << "[INFO] Found " << raw_paths.size() << " raw paths\n";

    std::sort(raw_paths.begin(), raw_paths.end(), [](const PathInfo& a, const PathInfo& b) {
        return a.ranking_score > b.ranking_score;
    });
    if (static_cast<int>(raw_paths.size()) > max_paths) {
        raw_paths.resize(max_paths);
    }

    std::cout << "[INFO] Scoring and ranking paths...\n";
    write_paths(output_file, raw_paths, detection_lookup);

    return 0;
}
