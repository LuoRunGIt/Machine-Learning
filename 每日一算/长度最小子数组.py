class Solution:
    def minSubArrayLen(self, target: int, nums: list[int]) -> int:
        if nums==None:
            return 0
        for i in range(1,len(nums)+1):
            for j in range(0,len(nums)-i+1):
                k=sum(nums[j:j+i])
                #print(k)
                if k>=target:
                    return i
                    break

        return 0
#暴力求解 n2次方


#最优解为滑动窗口
class Solution:
    def minSubArrayLen(self, target: int, nums: list[int]) -> int:
        left = 0
        right = -1
        Sum = 0
        res = float("inf")  # 表示无限大数

        for _ in range(len(nums)):
            right += 1
            Sum += nums[right]
            while Sum >= target:
                res = min(res, right - left + 1)
                if res == 1:
                    return 1
                Sum -= nums[left]
                left += 1
        # 所有数都求和，大了就缩小，但是始终保持最小值res

        return 0 if res == float("inf") else res