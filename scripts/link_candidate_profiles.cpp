#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Profile {
    std::string profile_id;
    std::string camera;
    std::string day;
    std::string zone_label;
    std::string size_label;
    std::string time_bucket_label;
    int detection_count = 0;
    int unique_frames = 0;
    double avg_bbox_area = 0.0;

    int day_num = 0;
    int size_num = 1;
    int time_mid_min = 0;
    double det_norm = 0.0;
    double frames_norm = 0.0;
    double area_norm = 0.0;
};

struct LinkRow {
    std::string profile_id_a;
    std::string camera_a;
    std::string day_a;
    std::string time_bucket_a;
    std::string zone_a;
    std::string size_a;
    int detection_count_a = 0;
    int unique_frames_a = 0;

    std::string profile_id_b;
    std::string camera_b;
    std::string day_b;
    std::string time_bucket_b;
    std::string zone_b;
    std::string size_b;
    int detection_count_b = 0;
    int unique_frames_b = 0;

    bool same_camera = false;
    bool cross_camera = true;
    bool same_day = true;

    double time_similarity = 0.0;
    double size_similarity = 0.0;
    double day_similarity = 0.0;
    double evidence_strength = 0.0;
    double candidate_link_score = 0.0;

    std::string link_strength;
    std::string link_explanation;
};

static std::unordered_map<std::string, int> DAY_TO_NUM = {
    {"day1", 1}, {"day2", 2}, {"day3", 3}, {"day4", 4},
    {"day5", 5}, {"day6", 6}, {"day7", 7}
};

static std::unordered_map<std::string, int> SIZE_TO_NUM = {
    {"small", 0}, {"medium", 1}, {"large", 2}, {"xlarge", 3}, {"xxlarge", 4}
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> result;
    std::string cur;
    bool in_quotes = false;

    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            result.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    result.push_back(cur);
    return result;
}

std::string csv_escape(const std::string& s) {
    if (s.find(',') != std::string::npos || s.find('"') != std::string::npos) {
        std::string out = "\"";
        for (char c : s) {
            if (c == '"') out += "\"\"";
            else out += c;
        }
        out += "\"";
        return out;
    }
    return s;
}

int parse_time_bucket_mid(const std::string& label) {
    // Example: 18:10-18:19
    auto dash = label.find('-');
    std::string start = label.substr(0, dash);
    std::string end = label.substr(dash + 1);

    int sh = std::stoi(start.substr(0, 2));
    int sm = std::stoi(start.substr(3, 2));
    int eh = std::stoi(end.substr(0, 2));
    int em = std::stoi(end.substr(3, 2));

    int start_min = sh * 60 + sm;
    int end_min = eh * 60 + em;
    return (start_min + end_min) / 2;
}

double size_similarity(int a, int b) {
    double diff = std::abs(a - b);
    return std::max(0.0, 1.0 - diff / 4.0);
}

double time_similarity(int a, int b, int max_diff_min) {
    double diff = std::abs(a - b);
    return std::max(0.0, 1.0 - diff / max_diff_min);
}

double day_similarity(int a, int b) {
    int diff = std::abs(a - b);
    if (diff == 0) return 1.0;
    if (diff == 1) return 0.6;
    return 0.0;
}

double evidence_strength(double det_norm, double frames_norm) {
    return 0.5 * det_norm + 0.5 * frames_norm;
}

std::string make_link_explanation(
    const Profile& a,
    const Profile& b,
    double score,
    const std::string& strength,
    bool same_day,
    bool cross_camera
) {
    std::ostringstream oss;
    oss << (strength == "strong" ? "Strong" :
            strength == "moderate" ? "Moderate" : "Weak")
        << " candidate link (" << std::fixed << std::setprecision(2) << score << "): "
        << (cross_camera ? "cross-camera" : "same-camera") << ", "
        << (same_day ? "same-day" : "cross-day") << ", "
        << "similar time windows (" << a.time_bucket_label << " vs " << b.time_bucket_label << "), "
        << "similar size groups (" << a.size_label << " vs " << b.size_label << ")";
    return oss.str();
}

bool score_pair(
    const Profile& a,
    const Profile& b,
    bool allow_same_camera,
    int max_size_diff,
    LinkRow& out
) {
    bool same_camera = (a.camera == b.camera);
    if (same_camera && !allow_same_camera) return false;

    bool same_day = (a.day == b.day);
    bool cross_camera = (a.camera != b.camera);

    int size_diff = std::abs(a.size_num - b.size_num);
    if (size_diff > max_size_diff) return false;

    double t_sim = time_similarity(a.time_mid_min, b.time_mid_min);
    if (t_sim <= 0.0) return false;

    double s_sim = size_similarity(a.size_num, b.size_num);
    double d_sim = day_similarity(a.day_num, b.day_num);
    double e_a = evidence_strength(a.det_norm, a.frames_norm);
    double e_b = evidence_strength(b.det_norm, b.frames_norm);
    double evidence = (e_a + e_b) / 2.0;

    double same_day_bonus = same_day ? 0.15 : 0.0;
    double cross_camera_bonus = cross_camera ? 0.15 : 0.0;

    double score =
        0.35 * t_sim +
        0.20 * s_sim +
        0.15 * d_sim +
        0.15 * evidence +
        same_day_bonus +
        cross_camera_bonus;

    if (score > 1.0) score = 1.0;

    std::string strength;
    if (score >= 0.80) strength = "strong";
    else if (score >= 0.65) strength = "moderate";
    else if (score >= 0.50) strength = "weak";
    else return false;

    out.profile_id_a = a.profile_id;
    out.camera_a = a.camera;
    out.day_a = a.day;
    out.time_bucket_a = a.time_bucket_label;
    out.zone_a = a.zone_label;
    out.size_a = a.size_label;
    out.detection_count_a = a.detection_count;
    out.unique_frames_a = a.unique_frames;

    out.profile_id_b = b.profile_id;
    out.camera_b = b.camera;
    out.day_b = b.day;
    out.time_bucket_b = b.time_bucket_label;
    out.zone_b = b.zone_label;
    out.size_b = b.size_label;
    out.detection_count_b = b.detection_count;
    out.unique_frames_b = b.unique_frames;

    out.same_camera = same_camera;
    out.cross_camera = cross_camera;
    out.same_day = same_day;
    out.time_similarity = t_sim;
    out.size_similarity = s_sim;
    out.day_similarity = d_sim;
    out.evidence_strength = evidence;
    out.candidate_link_score = score;
    out.link_strength = strength;
    out.link_explanation = make_link_explanation(a, b, score, strength, same_day, cross_camera);

    return true;
}

struct UnionFind {
    std::vector<int> parent;
    explicit UnionFind(int n) : parent(n) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    void unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra != rb) parent[rb] = ra;
    }
};

void write_links_csv(const std::string& path, const std::vector<LinkRow>& links) {
    std::ofstream out(path);
    out << "profile_id_a,camera_a,day_a,time_bucket_a,zone_a,size_a,detection_count_a,unique_frames_a,"
           "profile_id_b,camera_b,day_b,time_bucket_b,zone_b,size_b,detection_count_b,unique_frames_b,"
           "same_camera,cross_camera,same_day,time_similarity,size_similarity,day_similarity,"
           "evidence_strength,candidate_link_score,link_strength,link_explanation\n";

    for (const auto& r : links) {
        out
            << csv_escape(r.profile_id_a) << ','
            << csv_escape(r.camera_a) << ','
            << csv_escape(r.day_a) << ','
            << csv_escape(r.time_bucket_a) << ','
            << csv_escape(r.zone_a) << ','
            << csv_escape(r.size_a) << ','
            << r.detection_count_a << ','
            << r.unique_frames_a << ','
            << csv_escape(r.profile_id_b) << ','
            << csv_escape(r.camera_b) << ','
            << csv_escape(r.day_b) << ','
            << csv_escape(r.time_bucket_b) << ','
            << csv_escape(r.zone_b) << ','
            << csv_escape(r.size_b) << ','
            << r.detection_count_b << ','
            << r.unique_frames_b << ','
            << (r.same_camera ? "True" : "False") << ','
            << (r.cross_camera ? "True" : "False") << ','
            << (r.same_day ? "True" : "False") << ','
            << std::fixed << std::setprecision(4)
            << r.time_similarity << ','
            << r.size_similarity << ','
            << r.day_similarity << ','
            << r.evidence_strength << ','
            << r.candidate_link_score << ','
            << csv_escape(r.link_strength) << ','
            << csv_escape(r.link_explanation) << '\n';
    }
}

void write_groups_csv(
    const std::string& path,
    const std::vector<Profile>& profiles,
    const std::vector<LinkRow>& links,
    double group_threshold
) {
    std::unordered_map<std::string, int> id_to_idx;
    for (int i = 0; i < (int)profiles.size(); ++i) {
        id_to_idx[profiles[i].profile_id] = i;
    }

    UnionFind uf((int)profiles.size());
    for (const auto& link : links) {
        if (link.candidate_link_score >= group_threshold) {
            uf.unite(id_to_idx[link.profile_id_a], id_to_idx[link.profile_id_b]);
        }
    }

    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < (int)profiles.size(); ++i) {
        groups[uf.find(i)].push_back(i);
    }

    std::ofstream out(path);
    out << "candidate_group_id,profile_id,camera,day,time_bucket_label,size_label,detection_count,unique_frames\n";

    int group_num = 1;
    for (auto& kv : groups) {
        if (kv.second.size() <= 1) continue;

        std::ostringstream gid;
        gid << "candidate_group_" << std::setw(3) << std::setfill('0') << group_num++;
        for (int idx : kv.second) {
            const auto& p = profiles[idx];
            out
                << gid.str() << ','
                << csv_escape(p.profile_id) << ','
                << csv_escape(p.camera) << ','
                << csv_escape(p.day) << ','
                << csv_escape(p.time_bucket_label) << ','
                << csv_escape(p.size_label) << ','
                << p.detection_count << ','
                << p.unique_frames << '\n';
        }
    }
}

int main(int argc, char* argv[]) {
    std::string input = "data/summaries/pseudonymous_profiles.csv";
    std::string links_output = "data/summaries/candidate_profile_links.csv";
    std::string groups_output = "data/summaries/candidate_profile_groups.csv";

    int min_detection_count = 5;
    int min_unique_frames = 3;
    int max_size_diff = 1;
    double group_threshold = 0.65;
    bool allow_same_camera = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](std::string& v) { if (i + 1 < argc) v = argv[++i]; };
        auto next_int = [&](int& v) { if (i + 1 < argc) v = std::stoi(argv[++i]); };
        auto next_double = [&](double& v) { if (i + 1 < argc) v = std::stod(argv[++i]); };

        if (arg == "--input") next(input);
        else if (arg == "--links-output") next(links_output);
        else if (arg == "--groups-output") next(groups_output);
        else if (arg == "--min-detection-count") next_int(min_detection_count);
        else if (arg == "--min-unique-frames") next_int(min_unique_frames);
        else if (arg == "--max-size-diff") next_int(max_size_diff);
        else if (arg == "--group-threshold") next_double(group_threshold);
        else if (arg == "--allow-same-camera") allow_same_camera = true;
    }

    std::ifstream in(input);
    if (!in) {
        std::cerr << "Could not open input: " << input << "\n";
        return 1;
    }

    std::string header;
    std::getline(in, header);
    auto cols = split_csv_line(header);

    std::unordered_map<std::string, int> col_idx;
    for (int i = 0; i < (int)cols.size(); ++i) col_idx[cols[i]] = i;

    std::vector<Profile> profiles;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto fields = split_csv_line(line);
        if ((int)fields.size() < (int)cols.size()) continue;

        Profile p;
        p.profile_id = fields[col_idx["profile_id"]];
        p.camera = fields[col_idx["camera"]];
        p.day = fields[col_idx["day"]];
        p.zone_label = fields[col_idx["zone_label"]];
        p.size_label = fields[col_idx["size_label"]];
        p.time_bucket_label = fields[col_idx["time_bucket_label"]];
        p.detection_count = std::stoi(fields[col_idx["detection_count"]]);
        p.unique_frames = std::stoi(fields[col_idx["unique_frames"]]);
        p.avg_bbox_area = std::stod(fields[col_idx["avg_bbox_area"]]);

        if (p.detection_count < min_detection_count || p.unique_frames < min_unique_frames) {
            continue;
        }

        p.day_num = DAY_TO_NUM.count(p.day) ? DAY_TO_NUM[p.day] : 0;
        p.size_num = SIZE_TO_NUM.count(p.size_label) ? SIZE_TO_NUM[p.size_label] : 1;
        p.time_mid_min = parse_time_bucket_mid(p.time_bucket_label);

        profiles.push_back(p);
    }

    std::cout << "[INFO] Profiles after filtering: " << profiles.size() << "\n";
    if (profiles.empty()) {
        std::ofstream(links_output) << "";
        std::ofstream(groups_output) << "";
        return 0;
    }

    double max_det = 1.0, max_frames = 1.0, max_area = 1.0;
    for (const auto& p : profiles) {
        max_det = std::max(max_det, (double)p.detection_count);
        max_frames = std::max(max_frames, (double)p.unique_frames);
        max_area = std::max(max_area, p.avg_bbox_area);
    }
    for (auto& p : profiles) {
        p.det_norm = p.detection_count / max_det;
        p.frames_norm = p.unique_frames / max_frames;
        p.area_norm = p.avg_bbox_area / max_area;
    }

    std::sort(profiles.begin(), profiles.end(), [](const Profile& a, const Profile& b) {
        if (a.day_num != b.day_num) return a.day_num < b.day_num;
        if (a.time_mid_min != b.time_mid_min) return a.time_mid_min < b.time_mid_min;
        return a.profile_id < b.profile_id;
    });

    std::vector<LinkRow> links;
    auto start = std::chrono::steady_clock::now();
    long long compared = 0;
    long long kept = 0;

    for (int i = 0; i < (int)profiles.size(); ++i) {
        const auto& a = profiles[i];

        for (int j = i + 1; j < (int)profiles.size(); ++j) {
            const auto& b = profiles[j];

            if (b.day_num != a.day_num) break;

            compared++;

            LinkRow row;
            if (score_pair(a, b, allow_same_camera, max_size_diff, row)) {
                links.push_back(row);
                kept++;
            }
        }

        if ((i + 1) % 1000 == 0 || i + 1 == (int)profiles.size()) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double pct = 100.0 * (i + 1) / profiles.size();
            std::cout << "[PROGRESS] " << std::fixed << std::setprecision(1)
                      << pct << "% | profiles " << (i + 1) << "/" << profiles.size()
                      << " | compared " << compared
                      << " | kept " << kept
                      << " | elapsed " << elapsed / 60.0 << " min\n";
        }
    }

    std::sort(links.begin(), links.end(), [](const LinkRow& a, const LinkRow& b) {
        return a.candidate_link_score > b.candidate_link_score;
    });

    write_links_csv(links_output, links);
    write_groups_csv(groups_output, profiles, links, group_threshold);

    std::cout << "[OK] Wrote " << links_output << "\n";
    std::cout << "[OK] Wrote " << groups_output << "\n";
    std::cout << "[DONE] Candidate linking complete.\n";

    return 0;
}