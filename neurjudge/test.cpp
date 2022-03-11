#include <bits/stdc++.h>
using namespace std;

class Solution {
 public:
  int gcd(int a, int b) { return b ? gcd(b, a % b) : a; }
  long long countPairs(vector<int>& nums, int k) {
    int mx = k;
    for (auto i : nums) mx = max(mx, i);
    int cnt[mx + 1];
    memset(cnt, 0, sizeof cnt);
    for (auto i : nums) cnt[i]++;
    // cnt[i]：有多少个数字i的倍数出现
    for (int i = 1; i <= mx; i++)
      for (int j = i * 2; j <= mx; j += i) cnt[i] += cnt[j];

    // 这一步调和级数：n(1 + 1/2 + 1/3 + 1/4 + ...)
    // 调和级数最终收敛到logn

    long long res = 0;
    for (auto x : nums) {
      res += cnt[k / gcd(x, k)];
    }
    for (auto x : nums) {
      if (1ll * x * x % k == 0) res--;
    }
    return res / 2;
  }
};