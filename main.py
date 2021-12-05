import tkinter as tk
import pickle
from tkinter import font
from tkinter.messagebox import NO
from run import translate

class GUI():
    def __init__(self,state):
        if(state == 0):
            self.main_window()
        else:
            self.translation_inter()

    def translation_inter(self):
        def translation(event=None):
            org_s = b1.get().strip()
            trans_s = translate(org_s)
            text_box.delete(1.0, 'end')
            text_box.insert('insert', trans_s)

        def reset():
            text_box.delete(1.0,'end')

        self.window = tk.Tk()
        self.window.title('Translation (EN->ZH)')
        self.window.geometry('900x600')

        # 画布
        canvas = tk.Canvas(self.window, height=700, width=400)
        canvas.pack()

        # 单行文本
        L1 = tk.Label(self.window, text="Please input a sentence in English:", font=("Times New Roman", 12))
        L1.place(x=80, y=60)

        # 单行文本框, 外部输入
        b1 = tk.Entry(self.window, font=("Times New Roman", 12), show=None, width=85)
        b1.bind('<Return>', translation) # 绑定回车事件
        b1.place(x=80, y=100)

        # 设置查询按钮
        bt_click = tk.Button(self.window, text="Translate", width=25, height=2, command=translation, font=("Times New Roman", 10))
        bt_click.place(x=180, y=180)

        #设置清空翻译区按钮
        bt_trans = tk.Button(self.window, text="Clear the board",width=24,height=2,command=reset, font=("Times New Roman", 10))
        bt_trans.place(x=430, y=180)

        # 设置多行文本框 宽 高 文本框中字体 选中文字时文字的颜色
        text_box = tk.Text(self.window, width=113, height=20, font=("Times New Roman", 10))  # 显示多行文本
        text_box.place(x=80, y=260)

        # 进入消息循环
        self.window.mainloop()

    def main_window(self):
        self.window=tk.Tk()
        self.window.title('Welcome to the translation system (EN->ZH)')
        self.window.geometry('450x300')
        # 设置背景
        self.canvas=tk.Canvas(self.window,height=300,width=500)
        self.canvas.pack(side='top')
        # 标签 用户名密码
        tk.Label(self.window,text='user:', font=("Times New Roman", 12)).place(x=100,y=100)
        tk.Label(self.window,text='pwd:',font=("Times New Roman", 12)).place(x=100,y=150)
        # 用户名输入框
        self.var_usr_name=tk.StringVar()
        self.entry_usr_name=tk.Entry(self.window,textvariable=self.var_usr_name, width=20)
        self.entry_usr_name.place(x=160,y=100)
        # 密码输入框
        self.var_usr_pwd=tk.StringVar()
        self.entry_usr_pwd=tk.Entry(self.window,textvariable=self.var_usr_pwd,show='*', width=20)
        self.entry_usr_pwd.place(x=160,y=150)
        # 登录 注册按钮
        bt_login=tk.Button(self.window,text='Login',command=self.login, font=("Times New Roman", 10), width=10, height=2)
        bt_login.place(x=120,y=210)
        bt_logup=tk.Button(self.window,text='Register',command=self.register, font=("Times New Roman", 10), width=10, height=2)
        bt_logup.place(x=250,y=210)
        # bt_logquit=tk.Button(self.window,text='quit',command=quit)
        # bt_logquit.place(x=280,y=230)
        # 主循环
        self.window.mainloop()

    #登录函数
    def login(self):
        usr_name=self.var_usr_name.get()
        usr_pwd=self.var_usr_pwd.get()
        # 从本地字典获取用户信息，如果没有则新建本地数据库
        try:
            with open('usr_info.pickle','rb') as usr_file:
                usrs_info=pickle.load(usr_file)
        except FileNotFoundError:
            with open('usr_info.pickle','wb') as usr_file:
                usrs_info={'admin':'admin'}
                pickle.dump(usrs_info,usr_file)

        # 判断用户名和密码是否匹配
        if usr_name in usrs_info:
            if usr_pwd == usrs_info[usr_name]:
                tk.messagebox.showinfo(title='welcome', message='login successfully!')
                # 关闭登录界面
                self.window.destroy()
                gg=GUI(1)
                #self.translation_inter()
            else:
                tk.messagebox.showerror(message='password error!')
        # 用户名密码不能为空
        elif usr_name=='':
            tk.messagebox.showerror('wrong','user_name is empty!')
        elif  usr_pwd=='':
            tk.messagebox.showerror('wrong','user_pwd is empty!')
        # 不在数据库中弹出是否注册的框
        else:
            is_signup=tk.messagebox.askyesno('welcome','You have not registered yet, register now?')
            if is_signup:
                self.register()

    #注册函数
    def register(self):
        # 确认注册时的相应函数
        def getinfo():
            # 获取输入框内的内容
            reg_name=new_name.get()
            reg_pwd=new_pwd.get()
            reg_pwd2=new_pwd_confirm.get()#确认密码

            # 本地加载已有用户信息,如果没有则已有用户信息为空
            try:
                with open('usr_info.pickle','rb') as usr_file:
                    exist_usr_info=pickle.load(usr_file)
            except FileNotFoundError:
                exist_usr_info={}

            # 检查用户名存在、密码为空、密码前后不一致
            if reg_name in exist_usr_info:
                tk.messagebox.showerror('wrong','user name already exists!')
            elif reg_pwd =='':
                tk.messagebox.showerror('wrong','username is empty!')
            elif reg_pwd2=='':
                tk.messagebox.showerror('wrong','password is empty')
            elif reg_pwd !=reg_pwd2:
                tk.messagebox.showerror('wrong','passwords do not match!')
            # 注册信息没有问题则将用户名密码写入数据库
            else:
                exist_usr_info[reg_name]=reg_pwd
                with open('usr_info.pickle','wb') as usr_file:
                    pickle.dump(exist_usr_info,usr_file)
                tk.messagebox.showinfo('welcome','register successfully')
                # 注册成功关闭注册框
                window_sign_up.destroy()
                
        # 新建注册界面
        window_sign_up=tk.Toplevel(self.window)
        window_sign_up.geometry('350x200')
        window_sign_up.title('Register')
        # 用户名变量及标签、输入框
        new_name=tk.StringVar()
        tk.Label(window_sign_up,text='User name: ', font=("Times New Roman", 12)).place(x=10,y=10)
        tk.Entry(window_sign_up,textvariable=new_name).place(x=150,y=10)
        # 密码变量及标签、输入框
        new_pwd=tk.StringVar()
        tk.Label(window_sign_up,text='Input pwd: ', font=("Times New Roman", 12)).place(x=10,y=50)
        tk.Entry(window_sign_up,textvariable=new_pwd,show='*').place(x=150,y=50)
        # 重复密码变量及标签、输入框
        new_pwd_confirm=tk.StringVar()
        tk.Label(window_sign_up,text='Input pwd agagin: ', font=("Times New Roman", 12)).place(x=10,y=90)
        tk.Entry(window_sign_up,textvariable=new_pwd_confirm,show='*').place(x=150,y=90)
        # 确认注册按钮及位置
        bt_confirm_sign_up=tk.Button(window_sign_up,text='Register confirm',command=getinfo, font=("Times New Roman", 12))
        bt_confirm_sign_up.place(x=130,y=130)

    #退出主窗体
    def quit(self):
        self.window.destroy() #将窗体销毁

if __name__ == '__main__':
    gui=GUI(0)
