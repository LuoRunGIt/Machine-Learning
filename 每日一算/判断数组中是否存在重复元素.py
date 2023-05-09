class Solution:
    def containsDuplicate(self, nums: list[int]) -> bool:
        #dict表示的是一个字典
        hashtable = dict()
        for i, num in enumerate(nums):
            if num in hashtable:
                return True
            hashtable[num] = i
        return False
