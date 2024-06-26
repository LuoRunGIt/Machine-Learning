# 技能类
class Skill:
    def __init__(self):
        self.id = 0
        self.name = ""
        self.damage = 0
        # 3类 物理0、特殊1、变化2、物理aoe3、特殊aoe4、变化aoe5、打全场6、以此类推
        self.Class = 0
        # 命中率
        self.hit = 100
        # 命中率
        self.PP = 10
        # 属性
        self.attribute = "一般"
        self.description = "技能说明"
        #招式优先级
        self.priority=0
        #特殊类别如声音类
        self.special=0

    # 技能效果
    def skill_effects(self):
        print(self.description)

    # 设置技能基本情况
    def set_skill(self, num=0, name="未知", damage=0, classes=0, hit=100, PP=10, attibute="一般", description="",priority=0,special=0):
        self.id = num
        self.name = name
        self.damage = damage
        # 3类 物理0、特殊1、变化2
        self.Class = classes
        # 命中率
        self.hit = hit
        # 命中率
        self.PP = PP
        # 属性
        self.attribute = attibute
        self.description = description
        self.priority=priority
        self.special=special

