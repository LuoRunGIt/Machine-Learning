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

# 插入，未完成版，没有参数化
def insertSkill():
    db = connetDB.connect_db()
    # 创建游标对象cursor
    cursor = db.cursor()
    sql = '''
        INSERT INTO skills(name,damage,classes,hit,PP,attribute,description,priority) 
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s);'''
    cursor.execute(sql, ('地震', 100, 0, 100, 10, '地面', '除自身以外场上全部可以攻击到的宝可梦', 0))
    db.commit()#这句一定要有，提交事务
    db.close()


#insertSkill()
