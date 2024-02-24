
#装备类，其实类似于伤害类
class Equipment:
    def __init__(self):
        #装备id
        self.eq_id=0
        self.eq_name=""
        self.eq_desc="未装备道具"

    def equ_affect(self):
        print(self.eq_desc)

    def set_equ(self,id=0,name="",desc=""):
        self.eq_id = id
        self.eq_name = name
        self.eq_desc = desc
