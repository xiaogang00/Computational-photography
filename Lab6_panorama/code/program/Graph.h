#pragma once

#include <vector>
#include <stack>
#include <queue>
#include <ciso646>

template <typename EdgeDataType>
class Graph {
public:
  
  Graph();
  
  int addVertex(); // node i
  void addEdge(int from_node, int to_node, const EdgeDataType& data);
  
  template <typename Visitor>
  void dfs(int start_node, const Visitor& visitor);
  
private:
  template <typename Visitor>
  void _dfs(int node, int parent, int last_edge, const Visitor& visitor, std::vector<bool>& visited);
  
private:
  // Node i
  int m_NumVertices;
  std::vector<int> m_AdjacencyList; // j = m_AdjacencyList[i]
  std::vector<int> m_NodeList; // i' = m_NodeList[j];
  std::vector<int> m_SiblingList; // j' = m_SiblingList[j];
  std::vector<EdgeDataType> m_EdgeDataList;
  
};

template <typename EdgeDataType>
inline Graph<EdgeDataType>::Graph()
: m_NumVertices(0)
{
  m_AdjacencyList.reserve(64);
  m_NodeList.reserve(128);
  m_SiblingList.reserve(128);
  m_EdgeDataList.reserve(128);
}

template <typename EdgeDataType>
inline int Graph<EdgeDataType>::addVertex()
{
  int node = m_AdjacencyList.size();
  m_AdjacencyList.push_back(-1);
  return node;
}

template <typename EdgeDataType>
inline void Graph<EdgeDataType>::addEdge(int from_node, int to_node, const EdgeDataType& data)
{
  int edge = m_AdjacencyList[from_node];
  
  if (edge < 0) {
    
    edge = m_NodeList.size();
    m_NodeList.push_back(to_node);
    m_SiblingList.push_back(-1);
    m_EdgeDataList.push_back(data);
    m_AdjacencyList[from_node] = edge;
    
  } else {
    
    while (true) {
      int next_node = m_NodeList[edge];
      if (next_node == to_node)
        break;
      
      int next_edge = m_SiblingList[edge];
      if (next_edge == -1) {
        next_edge = m_NodeList.size();
        m_NodeList.push_back(to_node);
        m_SiblingList.push_back(-1);
        m_EdgeDataList.push_back(data);
        m_SiblingList[edge] = next_edge;
        break;
      } else {
        edge = next_edge;
      }
    }
    
  }
}

template <typename EdgeDataType>
template <typename Visitor>
inline void Graph<EdgeDataType>::dfs(int start_node, const Visitor& visitor)
{
  std::vector<bool> visited(m_AdjacencyList.size(), false);
  _dfs(start_node, -1, -1, visitor, visited);
}

template <typename EdgeDataType>
template <typename Visitor>
inline void Graph<EdgeDataType>::_dfs(int node, int parent, int last_edge, const Visitor& visitor, std::vector<bool>& visited)
{
  if (not visited[node]) {
    const EdgeDataType* edata = nullptr;
    if (last_edge != -1)
      edata = &m_EdgeDataList[last_edge];
    
    visitor(node, parent, edata);
    visited[node] = true;
    
    int edge = m_AdjacencyList[node];
    for ( ; edge >= 0 ; edge = m_SiblingList[edge] ) {
      _dfs(m_NodeList[edge], node, edge, visitor, visited);
    }
  }
}

