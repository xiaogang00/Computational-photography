#pragma once

#include <vector>

class DisjointSet {
public:
  DisjointSet();
  DisjointSet(int n);
  
  void reset(int num);
  int add();
  int find(int x);
  void merge(int x, int y);
  
private:
  std::vector<int> m_Parent;
  std::vector<int> m_Rank;
};

inline DisjointSet::DisjointSet()
{
}

inline DisjointSet::DisjointSet(int n)
: m_Parent(n), m_Rank(n)
{
  for (int i = 0; i < n; ++i) {
    m_Parent[i] = i;
    m_Rank[i] = 0;
  }
}

inline void DisjointSet::reset(int num)
{
  m_Parent.resize(num);
  m_Rank.resize(num);
  
  for (int i = 0; i < num; ++i) {
    m_Parent[i] = i;
    m_Rank[i] = 0;
  }
}

inline int DisjointSet::add()
{
  int idx = m_Parent.size();
  m_Parent.push_back(idx);
  m_Rank.push_back(0);
  return idx;
}

inline int DisjointSet::find(int x)
{
  int x_parent = m_Parent[x];
  
  if (x_parent != x) {
    x_parent = find(x_parent);
    m_Parent[x] = x_parent;
  }
  
  return x_parent;
}

inline void DisjointSet::merge(int x, int y)
{
  int xroot = find(x), yroot = find(y);
  if (xroot != yroot) {
    int xroot_rank = m_Rank[xroot];
    int yroot_rank = m_Rank[yroot];
    
    if (xroot_rank < yroot_rank) {
      m_Parent[xroot] = yroot;
    } else if (xroot_rank > yroot_rank) {
      m_Parent[yroot] = xroot;
    } else {
      m_Parent[yroot] = xroot;
      m_Rank[yroot] = xroot_rank + 1;
    }
  }
}

