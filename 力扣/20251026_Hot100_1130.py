from typing import List, Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class MinStack:
    """ 155.最小栈 """
    def __init__(self):
        self.stack = []
        self.minVal = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if len(self.minVal) == 0:
            self.minVal.append(val)
        else:
            self.minVal.append(min(self.minVal[-1], val))

    def pop(self) -> None:
        _ = self.stack.pop()
        _ = self.minVal.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minVal[-1]

class LRUCache:
    """ 146.LRU缓存 """
    def __init__(self, capacity: int):
        from collections import OrderedDict

        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.cache:
            self.cache.move_to_end(key)
        return self.cache.get(key, -1)

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            _ = self.cache.popitem(last=False)

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """ 1.两数之和 """
        for i in range(len(nums)):
            if target - nums[i] in nums[i + 1:]:
                return [i, nums.index(target - nums[i], i + 1)]
    
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """ 160.相交链表 """
        ## 不是快慢指针
        pA = headA
        pB = headB
        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        return pA   # or pB
    
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """ 15.三数之和 """
        res = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                _tmp = [nums[i], nums[l], nums[r]]
                _sum = sum(_tmp)
                if _sum == 0:
                    res.append(_tmp)
                    while l < r and nums[l + 1] == nums[l]:
                        l += 1
                    while l < r and nums[r - 1] == nums[r]:
                        r -= 1
                    l += 1
                    r -= 1
                elif _sum < 0:
                    l += 1
                else:
                    r -= 1
        return res
    
    def canJump(self, nums: List[int]) -> bool:
        """ 55.跳跃游戏 """
        maxReach = 0
        for i, n in enumerate(nums):
            if maxReach < i:
                return False
            maxReach = max(maxReach, i + n)
        return True
            
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """ 238.除自己以外数组的乘积 """
        n = len(nums)
        res = [0] * n
        
        tmp = 1
        for i in range(n):
            res[i] = tmp
            tmp *= nums[i]
        tmp = 1
        for i in range(n - 1, -1, -1):
            res[i] *= tmp
            tmp *= nums[i]
        return res
    
    def search(self, nums: List[int], target: int) -> int:
        """ 33.搜索旋转排序数组 """
        l, r = 0, len(nums) - 1
        while l <= r:       # 标准二分查找
            m = (l + r) // 2
            if nums[m] == target:
                return m
            elif nums[m] < nums[r]:             # [m, r]升序
                if nums[m] < target <= nums[r]: # 若target在[m, r]范围
                    l = m + 1
                else:                           # 若target不在[m, r]范围, 收缩查找范围
                    r = m - 1
            else:
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
        return -1
    
    def maxProfit(self, prices: List[int]) -> int:
        """ 121.买卖股票的最佳时机 """
        """
        一次买卖, 状态:
            0 不持有
            1 持有
        """
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]
        dp[0] = [0, -prices[0]]
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(-prices[i], dp[i - 1][1])
        return dp[-1][0]
    
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """ 739.每日温度 """
        ## 递减栈
        res = [0] * len(temperatures)
        stack = []
        for i, t in enumerate(temperatures):
            while stack and t > temperatures[stack[-1]]:    # 如果t相比stack中记录的升高了, 则stack中保存的索引要更新. while可以刷新整个stack, 只要不退出
                ind = stack.pop()
                res[ind] = i - ind
            stack.append(i)     # 体现递减栈--假设没执行到while里面, 则stack记录的温度索引在递减
        return res
    
    def sortColors(self, nums: List[int]) -> None:
        """
        75.颜色分类 \n
        Do not return anything, modify nums in-place instead.
        """
        ## 双指针
        pt0 = pt1 = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                nums[i], nums[pt1] = nums[pt1], nums[i]
                pt1 += 1
            if nums[i] == 0:
                nums[i], nums[pt0] = nums[pt0], nums[i]
                if pt0 < pt1:
                    nums[i], nums[pt1] = nums[pt1], nums[i]
                pt0 += 1
                pt1 += 1

        ## 冒泡
        # n = len(nums)
        # for i in range(n - 1):
        #     flag = False
        #     for j in range(n - 1 - i):
        #         if nums[j] > nums[j + 1]:
        #             nums[j], nums[j + 1] = nums[j + 1], nums[j]
        #             flag = True
        #     if not flag:
        #         break

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """ 234.回文链表 """
        def reverseList(head):
            """ 反转链表 """
            _pre = None
            while head:
                _next = head.next
                head.next = _pre        # 第一次执行到这时_pre=None, 链表与前面断开了
                _pre = head
                head = _next            # 当head是最后一个节点, 最后一次执行while, _pre成为第一个节点
            return _pre

        slow = fast = head
        while fast.next and fast.next.next:     # while退出后, fast指向第二段的最后一个节点, slow指向第一段的最后一个节点
            slow = slow.next
            fast = fast.next.next
        head2 = reverseList(slow.next)          # slow.next指向第二段的第一个节点, 执行反转
        while head and head2:                   # 即使长短不同, 都work
            if head.val != head2.val:
                return False
            head = head.next
            head2 = head2.next
        return True

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """ 238.除自身以外数组的乘积 """
        res = [0] * len(nums)
        
        _tmp = 1
        for i, n in enumerate(nums):
            res[i] = _tmp
            _tmp *= nums[i]
        _tmp = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= _tmp
            _tmp *= nums[i]
        return res
    
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        114.二叉树展开为链表 
        Do not return anything, modify root in-place instead.
        """
        cur = root
        while cur:
            if not cur.left:
                cur = cur.right
            else:
                tmp = cur.right
                cur.right = cur.left
                cur.left = None

                toRight = cur
                while toRight.right:
                    toRight = toRight.right
                toRight.right = tmp

                cur = cur.right

    def canPartition(self, nums: List[int]) -> bool:
        """ 416.分割等和子集 """
        ## 背包--动态规划
        if sum(nums) % 2:
            return False
        target = sum(nums) // 2
        dp = [0] * (target + 1)     # dp[j] 装满容量为j的背包 所得最大价值是多少
        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):    # 一维dp 背包倒序 避免覆盖前面的
                dp[j] = max(dp[j - nums[i]] + nums[i], dp[j])
        return dp[-1] == target
    
    def singleNumber(self, nums: List[int]) -> int:
        """ 136.只出现一次的数字 """
        ## 位运算
        res = nums.pop(0)
        while nums:
            res ^= nums.pop()
        return res

        ## 不够算法
        # nums.sort()
        # stack = []
        # while nums:
        #     n = nums.pop(0)
        #     if not stack or stack[-1] != n:
        #         stack.append(n)
        #     else:
        #         stack.pop()
        # return stack[-1]

    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 46.全排列 """
        ## 回溯三部曲
        res = []
        path = []
        used = [False] * len(nums)

        def backtrack():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if used[i] == True:
                    continue
                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()
                used[i] = False
        
        backtrack()
        return res
    
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """ 240.搜索二维矩阵II """
        ## 利用matrix的特性
        m, n = len(matrix), len(matrix[0])
        rowInd, colInd = 0, n - 1       # 起点
        while rowInd < m and colInd >= 0:
            val = matrix[rowInd][colInd]
            if val == target:
                return True
            elif val < target:
                rowInd += 1
            else:
                colInd -= 1
        return False

        ## 不够算法 没有利用matrix特性
        # def binarySearch(nums):
        #     l, r = 0, len(nums) - 1
        #     while l <= r:
        #         m = (l + r) // 2
        #         if nums[m] == target:
        #             return True
        #         elif nums[m] > target:
        #             r = m - 1
        #         else:
        #             l = m + 1
        #     return False
        
        # for row in matrix:
        #     if binarySearch(row):
        #         return True
        # return False

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        """ 739.每日温度 """
        ## 递减栈, 参考之前
        answers = [0] * len(temperatures)
        stack = []
        for i, t in enumerate(temperatures):
            while stack and t > temperatures[stack[-1]]:
                ind = stack.pop()
                answers[ind] = i - ind
            stack.append(i)     # 若没执行while, 就能理解这是递减栈了
        return answers
    
    def longestConsecutive(self, nums: List[int]) -> int:
        """ 128.最长连续序列 """
        ## 动规, 能做 但复杂度不符合要求
        # if len(nums) == 0:
        #     return 0
        # nums = list(set(nums))
        # nums.sort()
        # n = len(nums)
        # dp = [1] * n        # dp[i] 以nums[i]结尾的最长连续序列的长度
        # for i in range(1, n):
        #     if nums[i - 1] + 1 == nums[i]:
        #         dp[i] = dp[i - 1] + 1
        # return max(dp)

        ## 建议这个解法-->参考之前
        nums = set(nums)
        res = 0
        for dg in nums:
            if dg - 1 not in nums:
                _len = 1
                while dg + 1 in nums:
                    _len += 1
                    dg += 1
                res = max(res, _len)
        return res
    
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """ 207.课程表 """
        ## 还是参考之前, 忘了
        from collections import defaultdict

        res = 0
        i2indegree = [0] * numCourses
        p2i = defaultdict(set)
        for cur, pre in prerequisites:
            i2indegree[cur] += 1
            p2i[pre].add(cur)
        queue = [i for i, indegree in enumerate(i2indegree) if indegree == 0]

        while queue:
            cur = queue.pop(0)
            res += 1
            for ind in p2i[cur]:
                i2indegree[ind] = max(0, i2indegree[ind] - 1)
                if i2indegree[ind] == 0:
                    queue.append(ind)
        return res == numCourses
        
    def sortColors(self, nums: List[int]) -> None:
        """
        75.颜色分类 \n
        Do not return anything, modify nums in-place instead.
        """
        ## 一趟扫描
        pos0 = pos1 = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                nums[i], nums[pos1] = nums[pos1], nums[i]
                pos1 += 1
            
            if nums[i] == 0:
                nums[i], nums[pos0] = nums[pos0], nums[i]
                pos0 += 1
                if nums[i] == 1:
                    nums[i], nums[pos1] = nums[pos1], nums[i]
                pos1 += 1

        ## 冒泡
        # n = len(nums)
        # for i in range(n - 1):
        #     flag = False
        #     for j in range(n - 1 - i):
        #         if nums[j] > nums[j + 1]:
        #             nums[j], nums[j + 1] = nums[j + 1], nums[j]
        #             flag = True
        #     if not flag:
        #         break
        
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 617.合并二叉树 """
        from collections import deque

        if not (root1 and root2):
            return root1 or root2
        
        queue = deque([[root1, root2]])
        while queue:
            node1, node2 = queue.popleft()
            node1.val += node2.val
            if node1.left and node2.left:
                queue.append([node1.left, node2.left])
            if node1.right and node2.right:
                queue.append([node1.right, node2.right])
            if not node1.left and node2.left:
                node1.left = node2.left
            if not node1.right and node2.right:
                node1.right = node2.right
        return root1
    
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 148.排序链表 """
        ## 归排
        def mysplit(head, step):
            """ 将链表head划分为两段: 长度为step + 长度为剩下, 返回第二段的第一个节点 """
            for _ in range(step - 1):
                if not head:
                    return
                head = head.next
            if head:
                res = head.next
                head.next = None
                return res
            else:
                return
        
        def mymerge(h1, h2):
            cur = dummyHead = ListNode()
            while h1 and h2:
                if h1.val < h2.val:
                    cur.next = h1
                    h1 = h1.next
                else:
                    cur.next = h2
                    h2 = h2.next
                cur = cur.next
            cur.next = h1 or h2
            return dummyHead.next
        
        if not (head and head.next):
            return head

        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        
        dummyHead = ListNode(next=head)
        
        step = 1
        while step < n:
            ## 一次归排
            prev = dummyHead
            cur = prev.next
            while cur:
                left = cur
                right = mysplit(left, step)
                cur = mysplit(right, step)

                prev.next = mymerge(left, right)
                while prev.next:
                    prev = prev.next

            step *= 2
        return dummyHead.next
    
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        """ 399.除法求值 """
        graph = {}
        for [s, t], v in zip(equations, values):
            if s not in graph:
                graph[s] = {t: v}
            else:
                graph[s][t] = v
            if t not in graph:
                graph[t] = {s: 1 / v}
            else:
                graph[t][s] = 1 / v
        
        def dfs(s, t):
            if s not in graph:
                return -1
            elif s == t:
                return 1
            for node in graph[s].keys():
                if node == t:
                    return graph[s][t]
                elif node not in visited:
                    visited.add(node)
                    v = dfs(node, t)
                    if v != -1:
                        return v * graph[s][node]
            return -1

        res = []
        for s, t in queries:
            visited = set()
            v = dfs(s, t)
            res.append(v)
        return res
    
    ## --- 01.13 继续 ---
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """ 1.两数之和 """
        for i in range(len(nums)):
            if target - nums[i] in nums[i + 1:]:
                return [i, nums.index(target - nums[i], i + 1)]
            
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """ 49.字母异位词分组 """
        from collections import defaultdict

        res = defaultdict(list)
        for s in strs:
            k = ''.join(sorted(s))
            res[k].append(s)
        return list(res.values())
    
    def longestConsecutive(self, nums: List[int]) -> int:
        """ 128.最长连续序列 """
        nums = set(nums)
        res = 0
        for n in nums:
            if n - 1 not in nums:
                _len = 1
                while n + 1 in nums:
                    _len += 1
                    n += 1
                res = max(res, _len)
        return res
    
    def moveZeroes(self, nums: List[int]) -> None:
        """
        283.移动零 \n
        Do not return anything, modify nums in-place instead.
        """
        ## 移动非零元素. 若移动零元素不好做
        ind = 0     # 指定非零元素索引
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[ind] = nums[i]
                ind += 1
        for i in range(ind, len(nums)):
            nums[i] = 0

    def maxArea(self, height: List[int]) -> int:
        """ 11.盛最多水的容器 """
        ## 关键是寻找最大容器的过程: 宽度在减小，只有换掉“短板”才可能获得更大的面积。
        i, j = 0, len(height) - 1
        maxArea = abs(i - j) * min(height[i], height[j])
        while i < j:
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
            area = abs(i - j) * min(height[i], height[j])
            maxArea = max(maxArea, area)
        return area
    
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """ 15.三数之和 """
        ## 双指针 Tip:(这题用回溯不太行)
        ## 还是不会呀你
        # res = []
        # nums.sort()
        # for i in range(len(nums) - 2):
        #     if i > 0 and nums[i - 1] == nums[i]:
        #         continue
        #     l, r = i + 1, len(nums) - 1
        #     while l < r:
        #         _tmp = [nums[i], nums[l], nums[r]]
        #         _sum = sum(_tmp)
        #         if _sum == 0:
        #             res.append(_tmp[:])
        #             while l < r and nums[l] == nums[l + 1]:
        #                 l += 1
        #             l += 1
        #             while l < r and nums[r - 1] == nums[r]:
        #                 r -= 1
        #             r -= 1
        #         elif _sum < 0:
        #             l += 1
        #         else:
        #             r -= 1
        # return res

        ## Again
        res = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i - 1] == nums[i]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                _tmp = [nums[i], nums[l], nums[r]]
                _sum = sum(_tmp)
                if _sum == 0:
                    res.append(_tmp[:])
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    r -= 1
                elif _sum < 0:
                    l += 1
                else:
                    r -= 1
        return res
    
    def lengthOfLongestSubstring(self, s: str) -> int:
        """ 3.无重复字符的最长子串 \n
            要求连续 """
        ## 滑窗/双指针
        maxLength = 0
        for i in range(len(s)):
            seen = set()
            for j in range(i, len(s)):
                if s[j] in seen:
                    break
                seen.add(s[j])
            maxLength = max(maxLength, len(seen))
        return maxLength
    
    def findAnagrams(self, s: str, p: str) -> List[int]:
        """ 438.找到字符串中所有字母异位词 """
        ## 滑窗, 也不是那么简单
        res = []
        m = len(s)
        n = len(p)
        if m < n:
            return res
        
        s_cts = [0] * 26
        p_cts = [0] * 26
        for i in range(n):
            s_cts[ord(s[i]) - ord('a')] += 1
            p_cts[ord(p[i]) - ord('a')] += 1
        if s_cts == p_cts:      # ?
            res.append(0)

        for i in range(n, m):
            s_cts[ord(s[i]) - ord('a')] += 1
            s_cts[ord(s[i - n]) - ord('a')] -= 1
            if s_cts == p_cts:
                res.append(i - n + 1)
        return res
    
    def subarraySum(self, nums: List[int], k: int) -> int:
        """ 560.和为K的子数组 """
        ## 怎么出了个前缀和: 比如截止到索引j和为prefixSumA, 截止到i和为prefixSumB, 若prefixSumB-prefixSumA==k, 就找到了一个结果j
        from collections import defaultdict
        prefixSums = defaultdict(int)
        res = 0
        
        prefixSum = 0
        prefixSums[0] = 1
        for i in range(len(nums)):
            prefixSum += nums[i]
            res += prefixSums[prefixSum - k]
            prefixSums[prefixSum] += 1
        return res

    def maxSubArray(self, nums: List[int]) -> int:
        """ 53.最大子数组和 """
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
        return max(dp)
    
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """ 56.合并区间 """
        ## 回首答案, 简单优雅
        intervals.sort(key=lambda x: x[0])
        res = [intervals[0]]
        for inter in intervals[1:]:
            if res[-1][-1] < inter[0]:
                res.append(inter)
            else:
                res[-1][-1] = max(res[-1][-1], inter[-1])
        return res

    def rotate(self, nums: List[int], k: int) -> None:
        """
        189.轮转数组 \n
        Do not return anything, modify nums in-place instead.
        """
        def rev(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        
        n = len(nums)
        k %= n
        rev(0, n - 1)
        rev(0, k - 1)
        rev(k, n - 1)

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """ 238.除了自身以外数组的乘积 """
        ## 正着一遍 倒着一遍
        n = len(nums)
        res = [0] * n

        tmp = 1
        for i in range(n):
            res[i] = tmp
            tmp *= nums[i]
        tmp = 1
        for i in range(n - 1, -1, -1):
            res[i] *= tmp
            tmp *= nums[i]
        
        return res
    
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        73.矩阵置零 \n
        Do not return anything, modify matrix in-place instead.
        """
        rows = len(matrix)
        cols = len(matrix[0])
        zeroInds = []
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    zeroInds.append([i, j])
        for tmp in zeroInds:
            rowInd, colInd = tmp
            matrix[rowInd][:] = [0] * cols
            for i in range(rows):
                matrix[i][colInd] = 0
        
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """ 54.螺旋矩阵 """
        l, r = 0, len(matrix[0]) - 1
        t, b = 0, len(matrix) - 1
        res = []
        while True:
            for i in range(l, r + 1):
                res.append(matrix[t][i])
            t += 1
            if t > b:
                break
            
            for i in range(t, b + 1):
                res.append(matrix[i][r])
            r -= 1
            if r < l:
                break

            for i in range(r, l - 1, -1):
                res.append(matrix[b][i])
            b -= 1
            if b < t:
                break

            for i in range(b, t - 1, -1):
                res.append(matrix[i][l])
            l += 1
            if l > r:
                break
        return res
    
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        48.旋转图像 \n
        Do not return anything, modify matrix in-place instead.
        """
        ## 转置 + 行反转
        # for i in range(len(matrix)):
        #     for j in range(i):
        #         matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        # for i in range(len(matrix)):
        #     matrix[i][:] = matrix[i][::-1]

        ## 更极致的空间复杂度
        n = len(matrix)
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(n):
            l, r = 0, len(matrix[i])
            while l < r:
                matrix[i][l], matrix[i][r] = matrix[i][r], matrix[i][l]
                l += 1
                r -= 1
        
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """ 240.搜索二维矩阵II """
        """
        思路:
            初始在右上角, 当 val < target, 此时在行最右边嘛 那这行都放弃, 向下走
                        当 val > target, 此时在列最上边嘛 那这列都放弃, 向左走
            ...
            [注意: 上面'放弃', 是放弃的整行/列, 放弃了, 就不要再想了]
            此时处在矩阵任意位置, 上面的行 和 右边的列, 都是已经放弃的了, 只能往左/下移动了嘛, 当val < target, 向下走
                                                                                   当val > target, 向左走
            end
        """
        m = len(matrix)
        n = len(matrix[0])
        rowInd = 0
        colInd = n - 1
        while rowInd < m and colInd >= 0:
            val = matrix[rowInd][colInd]
            if val == target:
                return True
            elif val < target:
                rowInd += 1
            else:
                colInd -= 1
        return False
    
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 206.反转链表 """
        # pre = None
        # cur = head
        # while cur:
        #     post = cur.next
        #     cur.next = pre
        #     pre = cur
        #     cur = post
        # return pre

        ## 递归实现  PDD三面
        pre = None

        def backtrack(node):
            nonlocal pre
            if not node:
                return
            post = node.next
            node.next = pre
            pre = node
            backtrack(post)
        
        backtrack(head)
        return pre
    
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """ 160.相交链表 """
        curA = headA
        curB = headB
        while curA != curB:
            curA = curA.next if curA else headB     # 注意if条件, 要考虑到curA/curB都能到达最后一个节点
            curB = curB.next if curB else headA
        return curA     # or curB

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        """ 234.回文链表 """
        def reverse(node):
            """ 反转链表 """
            _pre = None
            while node:
                _post = node.next
                node.next = _pre
                _pre = node

                node = _post
            return _pre

        slow = fast = head
        while slow.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        head2 = reverse(slow.next)
        while head and head2:
            if head.val != head2.val:
                return False
            head = head.next
            head2 = head2.next
        return True

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        """ 141.环形链表 """
        ## 快慢指针, 个人觉得更优雅版快慢指针
        if not head:
            return False
        
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

        ## 简单, 但感觉不太优雅
        # seen = set()
        # cur = head
        # while cur:
        #     if cur in seen:
        #         return True
        #     seen.add(cur)
        #     cur = cur.next
        # return False

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 142.环形链表II """
        slow = fast = head
        while fast:         # 2种退出情况: 1.fast为空 2.快慢指针相遇
            slow = slow.next
            fast = fast.next
            if fast:
                fast = fast.next
            
            if slow == fast:
                break
        if not fast:
            return
        
        fast = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow
    
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """ 21.合并两个有序链表 """
        ## 优化空间
        if not (list1 and list2):
            return list1 or list2
        cur = head = ListNode()
        while list1 and list2:
            if list1.val < list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        cur.next = list1 or list2
        return head.next
        
        # if not (list1 and list2):
        #     return list1 or list2
        # cur = head = ListNode()
        # while list1 and list2:
        #     if list1.val < list2.val:
        #         cur.next = ListNode(list1.val)
        #         list1 = list1.next
        #     else:
        #         cur.next = ListNode(list2.val)
        #         list2 = list2.next
        #     cur = cur.next
        # cur.next = list1 or list2
        # return head.next

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """ 2.两数之和 """
        mt10 = False        # more than 10
        cur = head = ListNode()
        while l1 or l2:
            _sum = 0        # 当前位置的和
            if l1:
                _sum += l1.val
                l1 = l1.next
            if l2:
                _sum += l2.val
                l2 = l2.next
            if mt10:
                _sum += 1
            cur.next = ListNode(_sum % 10)
            cur = cur.next
            mt10 = _sum >= 10       # 若当前节点的和超过10, 留到下一位处理
        if mt10:
            cur.next = ListNode(1)
        return head.next
        
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """ 19.删除链表的倒数第N个节点 """
        ## 一趟扫描
        dummyHead = ListNode()
        dummyHead.next = head

        start = end = dummyHead
        for _ in range(n):      # start与end距离为n
            end = end.next
        
        prev = dummyHead
        while end:              # end走到末尾, 则start指向倒数第n个. 有可能end已None
            prev = start
            start = start.next
            end = end.next
        prev.next = start.next
        return dummyHead.next
    
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 24.两两交换链表中的节点 """
        dummyHead = ListNode()
        dummyHead.next = head

        prev = dummyHead
        cur = head
        while cur and cur.next:
            first, second = cur, cur.next
            prev.next = second
            first.next = second.next
            second.next = first
            
            prev = first
            cur = first.next
        return dummyHead.next
    
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        """ 138.随机链表的复制 """
        ## 哈希
        # 复制节点
        old2new = {}
        cur = head
        while cur:
            old2new[cur] = Node(cur.val)
            cur = cur.next
        # 复制指向
        cur = head
        while cur:
            old2new[cur].next = old2new.get(cur.next, None)
            old2new[cur].random = old2new.get(cur.random, None)
            cur = cur.next
        return old2new.get(head, None)

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 148.排序链表 """
        def mysplit(head, step):
            for _ in range(step - 1):
                if not head:
                    return
                head = head.next
            
            if not head:
                return
            else:
                res = head.next
                head.next = None
                return res
        
        def mymerge(h1, h2):
            cur = dummyHead = ListNode()
            while h1 and h2:
                if h1.val < h2.val:
                    cur.next = h1
                    h1 = h1.next
                else:
                    cur.next = h2
                    h2 = h2.next
                cur = cur.next
            cur.next = h1 or h2
            
            return dummyHead.next

        if not (head and head.next):
            return head
        
        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        
        dummyHead = ListNode(next=head)
        step = 1
        while step < n:
            prev = dummyHead
            cur = prev.next
            while cur:
                left = cur
                right = mysplit(left, step)
                cur = mysplit(right, step)
                prev.next = mymerge(left, right)
                while prev.next:
                    prev = prev.next
            step *= 2
        return dummyHead.next

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """ 94.二叉树的中序遍历 """
        ## 递归
        # res = []

        # def backtrack(node):
        #     if not node:
        #         return
        #     backtrack(node.left)
        #     res.append(node.val)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return res
        
        ## 迭代
        res = []

        stack = []
        cur = root
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left
            else:
                node = stack.pop()
                res.append(node.val)
                cur = node.right
        return res
    
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """ 104.二叉树的最大深度 """
        ## 递归
        # res = 0

        # def backtrack(node, depth):
        #     nonlocal res
        #     if not node:
        #         return
        #     if depth > res:
        #         res = depth
        #     backtrack(node.left, depth + 1)
        #     backtrack(node.right, depth + 1)
        
        # backtrack(root, 1)
        # return res
        
        ## 迭代
        if not root:
            return 0
        
        res = []
        queue = [root]
        while queue:
            num_level = len(queue)
            res_level = []
            for _ in range(num_level):
                node = queue.pop(0)
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(res_level[:])
        return len(res)

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """ 226.翻转二叉树 """
        ## 递归
        # def backtrack(node):
        #     if not node:
        #         return
        #     node.left, node.right = node.right, node.left
        #     backtrack(node.left)
        #     backtrack(node.right)
        
        # backtrack(root)
        # return root

        ## 迭代
        if not root:
            return
        
        stack = [root]
        while stack:
            node = stack.pop(0)
            node.left, node.right = node.right, node.left
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return root
    
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """ 101.对称二叉树 """
        ## 递归 后序
        # def backtrack(left, right):
        #     if not (left or right):
        #         return True
        #     elif not (left and right):
        #         return False
        #     elif left.val != right.val:
        #         return False
            
        #     res1 = backtrack(left.left, right.right)
        #     res2 = backtrack(left.right, right.left)
        #     return res1 and res2

        # return backtrack(root.left, root.right)

        ## 迭代
        stack = [[root.left, root.right]]
        while stack:
            left, right = stack.pop()
            if not (left or right):
                continue
            elif not (left and right):
                return False
            elif left.val != right.val:
                return False
            
            stack.append([left.left, right.right])
            stack.append([left.right, right.left])
        return True
    
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """ 543.二叉树的直径 \n
            不就是求 根节点深度/二叉树高度 嘛   \n
            不是, 非常不是!!! 确实会用到 根节点深度/二叉树高度 --> 任意两个节点 的最长路径 """
        ## 递归
        res = 0

        def backtrack(node):
            """ 以node为根节点的树的高度 """
            nonlocal res
            if not node:
                return 0
            l = backtrack(node.left)
            r = backtrack(node.right)
            res = max(res, l + r)
            return max(l, r) + 1

        backtrack(root)
        return res
    
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """ 102.二叉树的层序遍历 """
        if not root:
            return []
        
        res = []
        queue = [root]
        while queue:
            num_level = len(queue)
            res_level = []
            for _ in range(num_level):
                node = queue.pop(0)
                res_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(res_level[:])
        return res
    
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        """ 108.将有序数组转化为二叉搜索树 """
        def backtrack(nums):
            if len(nums) == 0:
                return
            
            l, r = 0, len(nums) - 1
            m = (l + r) // 2
            root = TreeNode(nums[m])
            root.left = backtrack(nums[l:m])
            root.right = backtrack(nums[m + 1:])
            return root
        
        return backtrack(nums)
    
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """ 98.验证二叉搜索树 """
        ## 二叉搜索树 !!! 中序遍历 严格递增 !!!
        # 递归 中序
        curVal = float('-inf')

        def backtrack(node):
            nonlocal curVal
            if not node:
                return True
            
            leftRes = backtrack(node.left)
            if curVal < node.val:
                curVal = node.val
            else:
                return False
            rightRes = backtrack(node.right)
            return leftRes and rightRes
        
        return backtrack(root)
    
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """ 207.课程表 \n
            面试遇到过 """
        from collections import defaultdict

        res = 0
        i2indegree = [0] * numCourses
        pre2i = defaultdict(set)
        
        # 初始化
        for i, pre in prerequisites:
            i2indegree[i] += 1
            pre2i[pre].add(i)
        
        queue = [i for i, indegree in enumerate(i2indegree) if indegree == 0]
        while queue:
            i = queue.pop(0)
            for ind in pre2i[i]:
                i2indegree[ind] = max(0, i2indegree[ind] - 1)
                if i2indegree[ind] == 0:
                    queue.append(ind)
            res += 1
        return res == numCourses
            

if __name__ == '__main__':
    sl = Solution()

    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    print(sl.spiralOrder(matrix))