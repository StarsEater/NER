import pymysql

class MysqlExec:
    def __init__(self,host = "192.168.3.228",user="root",\
                     password="root",database="deepwise_single_disease",\
                     port=3307,charset='utf8',table_name="Pathology"):
        self.db = pymysql.connect(host=host, user=user,password=password, database=database,port=port,charset=charset)
        self.table_name = table_name
    def get_sql_result(self,sql):
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result

    def update_sql(self,sql):
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            self.db.commit()
        except:
            self.db.rollback()
            print("commit error")
        finally:
            cursor.close()
    def get_sql_result_by_words(self,words):
        sql = "SELECT " + ",".join(words)+" From "+ self.table_name
        return self.get_sql_result(sql)

    def update_sql_by_words(self,name1,value1,name2,value2):
        sql = "UPDATE "+str(self.table_name)+" SET "+str(name1)+" = " + str(value1) + " WHERE "+str(name2)+" = "+str(value2)
        print(sql)
        # return self.update_sql(sql)
if __name__ == '__main__':
    # res = get_sql_result("SELECT id FROM Pathology")
    # "UPDATE Pathology SET stand_sign = '" + res + "' WHERE id = " + str(idx)
    # print(len(res),res[:10])
    sqlexec = MysqlExec()
    res= sqlexec.get_sql_result_by_words(["id","stand_sign"],"Pathology")
    print(res[:3])