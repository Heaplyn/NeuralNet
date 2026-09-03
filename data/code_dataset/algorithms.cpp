#include <vector>
#include <iostream>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <list>
#include <memory>

namespace algorithms {

// --- 1. QuickSort with Median-of-Three Partitioning ---
template <typename T>
size_t partition(std::vector<T>& arr, int low, int high) {
    int mid = low + (high - low) / 2;
    if (arr[mid] < arr[low]) std::swap(arr[low], arr[mid]);
    if (arr[high] < arr[low]) std::swap(arr[low], arr[high]);
    if (arr[high] < arr[mid]) std::swap(arr[mid], arr[high]);
    
    T pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return static_cast<size_t>(i + 1);
}

template <typename T>
void quick_sort(std::vector<T>& arr, int low, int high) {
    if (low < high) {
        size_t pi = partition(arr, low, high);
        if (pi > 0) quick_sort(arr, low, static_cast<int>(pi - 1));
        quick_sort(arr, static_cast<int>(pi + 1), high);
    }
}

// --- 2. High-Performance LRU Cache ---
template <typename Key, typename Value>
class LRUCache {
private:
    size_t capacity;
    std::list<std::pair<Key, Value>> items;
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator> lookup;

public:
    explicit LRUCache(size_t cap) : capacity(cap) {}

    bool get(const Key& key, Value& out_val) {
        auto it = lookup.find(key);
        if (it == lookup.end()) return false;
        items.splice(items.begin(), items, it->second);
        out_val = it->second->second;
        return true;
    }

    void put(const Key& key, const Value& val) {
        auto it = lookup.find(key);
        if (it != lookup.end()) {
            it->second->second = val;
            items.splice(items.begin(), items, it->second);
            return;
        }
        if (items.size() >= capacity) {
            auto last = items.back();
            lookup.erase(last.first);
            items.pop_back();
        }
        items.emplace_front(key, val);
        lookup[key] = items.begin();
    }
};

// --- 3. Dijkstra Shortest Path on Adjacency List Graph ---
struct Edge {
    int to;
    float weight;
};

using Graph = std::vector<std::vector<Edge>>;

std::vector<float> dijkstra(const Graph& graph, int source) {
    size_t n = graph.size();
    std::vector<float> dist(n, 1e9f);
    dist[source] = 0.0f;

    using State = std::pair<float, int>;
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
    pq.push({0.0f, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;

        for (const auto& edge : graph[u]) {
            if (dist[u] + edge.weight < dist[edge.to]) {
                dist[edge.to] = dist[u] + edge.weight;
                pq.push({dist[edge.to], edge.to});
            }
        }
    }
    return dist;
}

// --- 4. Binary Search Tree (BST) ---
template <typename T>
class BinarySearchTree {
    struct Node {
        T value;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        explicit Node(T v) : value(v), left(nullptr), right(nullptr) {}
    };
    std::unique_ptr<Node> root;

    void insert_rec(std::unique_ptr<Node>& node, T val) {
        if (!node) {
            node = std::make_unique<Node>(val);
        } else if (val < node->value) {
            insert_rec(node->left, val);
        } else {
            insert_rec(node->right, val);
        }
    }

    bool contains_rec(const std::unique_ptr<Node>& node, T val) const {
        if (!node) return false;
        if (node->value == val) return true;
        if (val < node->value) return contains_rec(node->left, val);
        return contains_rec(node->right, val);
    }

public:
    void insert(T val) { insert_rec(root, val); }
    bool contains(T val) const { return contains_rec(root, val); }
};

} // namespace algorithms
