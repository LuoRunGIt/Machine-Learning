import string
class Solution:
    def isPalindrome(self, s: str) -> bool:
        if s==None:
            return True
        s=s.lower()#全部转小写
        s=s.translate(s.maketrans('', '', string.punctuation))#移除标点
        b=s.split()#先拆分，再合并
        s="".join(b)
        #print(c)
        right=len(s)-1
        left=0
        while(left<right):
            if s[left]==s[right]:
                left=left+1
                right=right-1
            else:
                return False
        return True