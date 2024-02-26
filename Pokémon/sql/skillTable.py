import connetDB
import pymysql


def createTable():
    try:
        db = connetDB.connect_db()
        # 创建游标对象cursor
        cursor = db.cursor()
        # 使用execute()方法执行sql，如果表存在则删除,算了不删了
        cursor.execute('drop table if EXISTS skills')
        # 创建表的sql
        sql = '''
            create table skills(
            id int primary key auto_increment,
            name varchar(50) not null,
            damage int not null,
            classes int not null,
            hit int not null,
            PP int not null,
            attribute varchar(20),
            description varchar(500),
            priority int not null
             )
        '''

        cursor.execute(sql)

    except:
        print('创建表失败')
    finally:
        # 关闭数据库连接
        db.close()


# 已经创建再运行就会报错createTable()

# 插入，未完成版，没有参数化，参数化8个参数
# 名称name,伤害damage,类别classes,命中率hit,使用次数PP,属性attribute,描述description,优先级priority
def insert_skill(name="", damage=0, classes=0, hit=100, pp=0, attribute="", description="", priority=0):
    db = connetDB.connect_db()
    # 创建游标对象cursor
    cursor = db.cursor()
    sql = '''
        INSERT INTO skills(name,damage,classes,hit,PP,attribute,description,priority) 
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s);'''
    # cursor.execute(sql, ('地震', 100, 0, 100, 10, '地面', '除自身以外场上全部可以攻击到的宝可梦', 0))
    cursor.execute(sql, (name, damage, classes, hit, pp, attribute, description, priority))
    db.commit()  # 这句一定要有，提交事务
    db.close()


#insert_skill("地震", 100, 0, 100, 10, "地面", "除自身以外场上全部可以攻击到的宝可梦", 0)
# insert_skill("电光束", 130, 1, 100, 10, "电", "第１回合收集电力提高特攻，第２回合将高压的电力发射出去。下雨天气时能立刻发射。", 0)

# 批量插入数据
# list格式 [("电光束", 130, 1, 100, 10, "电", "第１回合收集电力提高特攻，第２回合将高压的电力发射出去。下雨天气时能立刻发射。", 0)]
def insert_skills(skills_list):
    if skills_list == None:
        print("列表为空")
        return
    db = connetDB.connect_db()
    cursor = db.cursor()
    sql = '''
            INSERT INTO skills(name,damage,classes,hit,PP,attribute,description,priority) 
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s);'''

    try:
        # 执行sql语句
        cursor.executemany(sql, skills_list)
        # 提交事务
        db.commit()
        print('插入多条数据成功')
    except Exception as e:
        print(e)
        # 如果出现异常，回滚
        db.rollback()
        print('插入多条数据失败')
    finally:
        # 关闭数据库连接
        db.close()


'''
list=[("月亮之力",95,1,100,15,"妖精","借用月亮的力量攻击对手。有时会降低对手的特攻。",0),
      ("魔法闪耀",80,4,100,10,"妖精","妖精系aoe",0)]

insert_skills(list)
'''


def update_skill_damage(name="", damage=0):
    if name == "":
        print("不知道名字改什么改")
        return

    db = connetDB.connect_db()
    cursor = db.cursor()  # 游标
    sql = 'update skills set damage=%s where name=%s'
    try:
        cursor.execute(sql, (damage, name))
        db.commit()
        print('修改成功')
    except:
        print('修改失败')
        db.rollback()
    finally:
        db.close()


#update_skill_damage("地震",100)


def delete_skill(name=""):
    if name == "":
        print("不知道名字删什么删")
        return

    db = connetDB.connect_db()
    cursor = db.cursor()  # 游标
    sql = 'delete from skills where name=%s'
    try:
        cursor.execute(sql, (name))
        db.commit()
        print('删除成功')
    except:
        print('删除失败')
        db.rollback()
    finally:
        db.close()

#delete_skill(name="地震")