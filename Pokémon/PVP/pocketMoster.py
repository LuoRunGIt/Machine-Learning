# 宝可梦类
import math
import equipment

class pocketMoster:
    natures = {"怕寂寞": 0, "加攻击减防御": 0, "顽皮": 1, "加攻击减特防": 1, "勇敢": 2, "加攻击减速度": 2, "固执": 3, "加攻击减特攻": 3,
               "慢吞吞": 4, "加特攻减防御": 4, "内敛": 5, "加特攻减攻击": 5, "马虎": 6, "加特攻减特防": 6, "冷静": 7, "加特攻减速度": 7,
               "温顺": 8, "加特防减防御": 8, "温和": 9, "加特防减攻击": 9, "自大": 10, "加特防减速度": 10, "慎重": 11, "加特防减特攻": 11,
               "急躁": 12, "加速度减防御": 12, "胆小": 13, "加速度减攻击": 13, "天真": 14, "加速度减特防": 14, "爽朗": 15, "加速度减特攻": 15,
               "大胆": 16, "加防御减攻击": 16, "淘气": 17, "加防御减特攻": 17, "乐天": 18, "加防御减特防": 18, "悠闲": 19, "加防御减速度": 19,
               "勤奋": 20, "坦率": 20, "认真": 20, "害羞": 20, "浮躁": 20, "不加不减": 20
               }

    def __init__(self):
        # 常规名称
        self.name1 = ""
        # 预留的查询名称，这里可以是英文也可以是拼音，主要是为了后期在数据库中的查询
        self.name2 = ""
        # 等级默认50
        self.level = 50
        # 属性1、2
        self.attribute1 = ""
        self.attribute2 = ""

        # 编号，或许比较重要但是现在其实一般般
        num = 0
        # 特性,除了人马一体这种麻烦的，都是一个就行
        self.ability = ""
        # 性别,1雄性0雌性2无性别，其实性别对对战影响不大，但存在同编号但性别形态不同的宝可梦如：爱管侍
        sex = 1
        # 6项种族值
        self.b_Hp = 0
        self.b_Atk = 0
        self.b_Def = 0
        self.b_SpA = 0
        self.b_SpD = 0
        self.b_Spe = 0
        # 6项努力值
        self.EVs_Hp = 0
        self.EVs_Atk = 0
        self.EVs_Def = 0
        self.EVs_SpA = 0
        self.EVs_SpD = 0
        self.EVs_Spe = 0

        # 个体值，默认满个体
        self.i_Hp = 31
        self.i_Atk = 31
        self.i_Def = 31
        self.i_SpA = 31
        self.i_SpD = 31
        self.i_Spe = 31
        # 性格修正
        self.nature = "勤奋"
        self.F_Hp = 31
        self.F_Atk = 31
        self.F_Def = 31
        self.F_SpA = 31
        self.F_SpD = 31
        self.F_Spe = 31
        self.equ=equipment.Equipment

    # 最终计算出6维
    def Panel(self):
        # 面板属性
        self.F_Hp = math.floor(
            (self.b_Hp * 2 + self.i_Hp + math.floor(self.EVs_Hp / 4)) * self.level / 100) + 10 + self.level
        self.F_Atk = math.floor((self.b_Atk * 2 + self.i_Atk + math.floor(self.EVs_Atk / 4)) * self.level / 100) + 5
        self.F_Def = math.floor((self.b_Def * 2 + self.i_Def + math.floor(self.EVs_Def / 4)) * self.level / 100) + 5
        self.F_SpA = math.floor((self.b_SpA * 2 + self.i_SpA + math.floor(self.EVs_SpA / 4)) * self.level / 100) + 5
        self.F_SpD = math.floor((self.b_SpD * 2 + self.i_SpD + math.floor(self.EVs_SpD / 4)) * self.level / 100) + 5
        self.F_Spe = math.floor((self.b_Spe * 2 + self.i_Spe + math.floor(self.EVs_Spe / 4)) * self.level / 100) + 5

        # 不变
        if self.natures[self.nature] == 20:
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加防御
        # 加防御减速度
        if self.natures[self.nature] == 19:
            self.F_Def += math.floor(self.F_Def * 0.1)
            self.F_Spe -= math.floor(self.F_Spe * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加防御减特防
        if self.natures[self.nature] == 18:
            self.F_Def += math.floor(self.F_Def * 0.1)
            self.F_SpD -= math.floor(self.F_SpD * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加防御减特攻
        if self.natures[self.nature] == 17:
            self.F_Def += math.floor(self.F_Def * 0.1)
            self.F_SpA -= math.floor(self.F_SpA * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加防御减攻击
        if self.natures[self.nature] == 16:
            self.F_Def += math.floor(self.F_Def * 0.1)
            self.F_Atk -= math.floor(self.F_Atk * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加速度
        # 加速度减特攻
        if self.natures[self.nature] == 15:
            self.F_Spe += math.floor(self.F_Spe * 0.1)
            self.F_SpA -= math.floor(self.F_SpA * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加速度减特防
        if self.natures[self.nature] == 14:
            self.F_Spe += math.floor(self.F_Spe * 0.1)
            self.F_SpD -= math.floor(self.F_SpD * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加速度减攻击
        if self.natures[self.nature] == 13:
            self.F_Spe += math.floor(self.F_Spe * 0.1)
            self.F_Atk -= math.floor(self.F_Atk * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加速度减防御
        if self.natures[self.nature] == 12:
            self.F_Spe += math.floor(self.F_Spe * 0.1)
            self.F_Def -= math.floor(self.F_Def * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加特防减
        # 加特防减特攻
        if self.natures[self.nature] == 11:
            self.F_SpD += math.floor(self.F_SpD * 0.1)
            self.F_SpA -= math.floor(self.F_SpA * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加特防减速度
        if self.natures[self.nature] == 10:
            self.F_SpD += math.floor(self.F_SpD * 0.1)
            self.F_Spe -= math.floor(self.F_Spe * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加特防减攻击
        if self.natures[self.nature] == 9:
            self.F_SpD += math.floor(self.F_SpD * 0.1)
            self.F_Atk -= math.floor(self.F_Atk * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加特防减防御
        if self.natures[self.nature] == 8:
            self.F_SpD += math.floor(self.F_SpD * 0.1)
            self.F_Def -= math.floor(self.F_Def * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加特攻减
        # 加特攻减速度
        if self.natures[self.nature] == 7:
            self.F_SpA += math.floor(self.F_SpA * 0.1)
            self.F_Spe -= math.floor(self.F_Spe * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加特攻减特防
        if self.natures[self.nature] == 6:
            self.F_SpA += math.floor(self.F_SpA * 0.1)
            self.F_SpD -= math.floor(self.F_SpD * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加特攻减攻击
        if self.natures[self.nature] == 5:
            self.F_SpA += math.floor(self.F_SpA * 0.1)
            self.F_Atk -= math.floor(self.F_Atk * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加特攻减防御
        if self.natures[self.nature] == 4:
            self.F_SpA += math.floor(self.F_SpA * 0.1)
            self.F_Def -= math.floor(self.F_Def * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加攻击减
        # 加攻击减特攻
        if self.natures[self.nature] == 3:
            self.F_Atk += math.floor(self.F_Atk * 0.1)
            self.F_SpA -= math.floor(self.F_SpA * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加攻击减速度
        if self.natures[self.nature] == 2:
            self.F_Atk += math.floor(self.F_Atk * 0.1)
            self.F_Spe -= math.floor(self.F_Spe * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加攻击减特防
        if self.natures[self.nature] == 1:
            self.F_Atk += math.floor(self.F_Atk * 0.1)
            self.F_SpD -= math.floor(self.F_SpD * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
        # 加攻击减防御
        if self.natures[self.nature] == 0:
            self.F_Atk += math.floor(self.F_Atk * 0.1)
            self.F_Def -= math.floor(self.F_Def * 0.1)
            return self.F_Hp, self.F_Atk, self.F_Def, self.F_SpA, self.F_SpD, self.F_Spe
    # 检查努力值分配是否存在错误
    def Evs_check(self):
        if self.EVs_Hp < 0 or self.EVs_Hp > 252:
            print("生命努力值错误")
            return -1
        if self.EVs_Atk < 0 or self.EVs_Atk > 252:
            print("攻击努力值错误")
            return -1
        if self.EVs_Def < 0 or self.EVs_Def > 252:
            print("防御努力值错误")
            return -1
        if self.EVs_SpA < 0 or self.EVs_SpA > 252:
            print("特攻努力值错误")
            return -1
        if self.EVs_SpD < 0 or self.EVs_SpD > 252:
            print("特防努力值错误")
            return -1
        if self.EVs_Spe < 0 or self.EVs_Spe > 252:
            print("速度努力值错误")
            return -1
        check = self.EVs_Hp + self.EVs_Atk + self.EVs_Def + self.EVs_SpA + self.EVs_SpD + self.EVs_Spe
        if check < 0 or check > 510:
            print("努力值溢出错误")
            return 0
        elif check < 510:
            print("努力值未分完")
            return 0
        else:
            print("努力值分完了")
            return 1
        return 2

    # 设置努力值
    def set_Evs(self, hp, atk, defense, spa, spd, spe):
        self.EVs_Hp = hp
        self.EVs_Atk = atk
        self.EVs_Def = defense
        self.EVs_SpA = spa
        self.EVs_SpD = spd
        self.EVs_Spe = spe
        if self.Evs_check() != 1:
            self.EVs_Hp = 0
            self.EVs_Atk = 0
            self.EVs_Def = 0
            self.EVs_SpA = 0
            self.EVs_SpD = 0
            self.EVs_Spe = 0

    # 设置性格修正
    def set_Nature(self, str):
        if str not in self.natures:
            print("未识别的性格修正")
            self.nature = "勤奋"
        else:
            self.nature = str

    def get_Nature(self):
        return self.nature

#测试函数
def test():
    # 满级斗笠菇未修正252攻击 359 252速度239 防御178 hp261
    #对照测试网站https://professorsidon.github.io/VGC-Damage-Calculator-Chinese/
    mypok = pocketMoster()
    mypok.b_Hp = 60
    mypok.b_Atk = 130
    mypok.b_Def = 80
    mypok.b_SpA = 60
    mypok.b_SpD = 60
    mypok.b_Spe = 70

    mypok.set_Nature("怕")
    mypok.set_Nature("爽朗")
    print(mypok.nature)

    mypok.set_Evs(6, 252, 0, 0, 0, 252)
    mypok.Evs_check()
    mypok.level = 50
    print(mypok.Panel())
