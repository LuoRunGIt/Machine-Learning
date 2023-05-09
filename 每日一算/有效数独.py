# a = [1, 4, 7]
for i in ([1, 4, 7]):
    for j in (1, 4, 7):
        print(i, j)
# 注意这里遍历数组的时候没有range
nums = []
for i in range(0, 10):
    nums.append(i)
print(nums)

broad = [[".", ".", ".", ".", "5", ".", ".", "1", "."], [".", "4", ".", "3", ".", ".", ".", ".", "."],
         [".", ".", ".", ".", ".", "3", ".", ".", "1"], ["8", ".", ".", ".", ".", ".", ".", "2", "."],
         [".", ".", "2", ".", "7", ".", ".", ".", "."], [".", "1", "5", ".", ".", ".", ".", ".", "."],
         [".", ".", ".", ".", ".", "2", ".", ".", "."], [".", "2", ".", "9", ".", ".", ".", ".", "."],
         [".", ".", "4", ".", ".", ".", ".", ".", "."]]
print(len(broad))
for data in range(len(broad)):
    print(broad[data])


class Solution:
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        # 我将其看作为2个部分组成
        # 一部分是每行，每列的比较
        # 一部分是9个9宫格进行比较。
        # hashtable=dict()
        result = True
        for i in ([1, 4, 7]):
            for j in ([1, 4, 7]):
                #  print(i,j)
                nums = [board[i - 1][j - 1], board[i][j - 1], board[i + 1][j - 1], board[i - 1][j], board[i][j],
                        board[i + 1][j], board[i - 1][j + 1], board[i][j + 1], board[i + 1][j + 1]]
                result = containsDuplicate(nums)
                # print(result)
                if result == True:
                    return False
                    break

        for i in range(0, 9):
            nums = []
            for j in range(0, 9):
                nums.append(board[i][j])
            result = containsDuplicate(nums)
            if result == True:
                return False
                break
            del (nums)
        for i in range(0, 9):
            nums = []
            for j in range(0, 9):
                nums.append(board[j][i])
            result = containsDuplicate(nums)
            if result == True:
                return False
                break
            # del(nums)
        return True


# 如果哈希表里有则表示有相同结果不成立
def containsDuplicate(strs: list[str]) -> bool:
    hashtable = dict()
    for i, num in enumerate(strs):
        if num in hashtable and num != '.':
            return True
        hashtable[num] = i
    return False
