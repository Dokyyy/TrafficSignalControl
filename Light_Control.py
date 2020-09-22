import os
import tkinter
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk

# global STOP # 用来提醒control端的服务关闭
# STOP = 0

def version():
    info = '''
优化：
1. 与ONENET云平台相连接 
2. 分离了sumo端与控制端'''
    tkinter.messagebox.showinfo('version1.0', info)

def exit_sc():
    tkinter.messagebox.showinfo('STOP', '终止运行......')
    stop_sc = 'taskkill /f /t /IM python.exe'
    os.popen(stop_sc)
    sc.destroy()

# 关于
def about():
    info = '''
这是一个基于交通流量的交通信号控制系统，
SUMO端模拟车流并将车流数据上传至ONENET
云平台，控制端获取车流信息并生成信号灯
控制信号，通过ONENET云平台下发至SUMO端，
改变信号灯相位以控制车流，缓解交通拥堵'''
    tkinter.messagebox.showinfo('关于', info)

# 使用方法
def use():
    info = '''
点击SUMO启动SUMO模拟端
点击CONTROL启动控制端
点击EXTI或菜单栏<退出>关闭系统'''
    tkinter.messagebox.showinfo('使用方法', info)

# 运行SUMO端
def SUMO():
    run_sumo = r'python sumo\runexp_colight.py'
    tkinter.messagebox.showinfo('SUMO', '启动sumo端......')
    os.popen(run_sumo)

# 运行控制端
def CONTROL():
    # run_control = r'D:\Anaconda3\envs\tensorflow\python.exe E:\python\Py3\light_control\control\Control.py'
    run_control = r'D:\Anaconda3\envs\tensorflow\python.exe control\Control.py'
    tkinter.messagebox.showinfo('CONTROL', '启动控制端......')
    # del_history_file = 'rmdir /s/q cloud_data'
    # os.popen(del_history_file)
    os.popen(run_control)


if __name__ == '__main__':
    # 初始化一个窗体对象
    sc = Tk()
    sc.title('基于车流的交通信号控制系统')

    image = Image.open(r'img.jpg')
    background_image = ImageTk.PhotoImage(image)
    w = background_image.width()
    h = background_image.height()
    sw = sc.winfo_screenwidth()
    sh = sc.winfo_screenheight()
    sc.geometry('%dx%d+%d+%d' % (w, h, (sw - w) / 2, (sh - h) / 2))
    background_label = Label(sc, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # tkinter.messagebox.showinfo('欢迎','欢迎使用基于车流的交通信号控制平台')

    # 窗体内容
    label_title = Label(sc, text='交通信号控制平台', bg='cyan',
                        font=('黑体', 14, 'bold'), width=5, height=2)
    label_title.pack(fill=X)

    menubar = Menu(sc)
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label='版本信息', command=version)
    filemenu.add_separator()
    filemenu.add_command(label='退出', command=exit_sc)
    menubar.add_cascade(label='菜单', menu=filemenu)

    helpmenu = Menu(menubar, tearoff=0)
    helpmenu.add_command(label='关于', command=about)
    helpmenu.add_separator()
    helpmenu.add_command(label='使用方法', command=use)
    menubar.add_cascade(label='帮助', menu=helpmenu)
    sc.config(menu=menubar)

    buttons = Button(sc, text='SUMO', command=SUMO, width=10, height=1, fg='green',
                     font=('Consolas', 10, 'bold'))
    buttons.place(anchor=CENTER, x=440, y=130)

    buttons = Button(sc, text='CONTROL', command=CONTROL, width=10, height=1, fg='green',
                     font=('Consolas', 10, 'bold'))
    buttons.place(anchor=CENTER, x=440, y=180)

    buttone = Button(sc, text='EXIT', command=exit_sc, width=10, height=1, fg='red',
                     font=('Consolas', 10, 'bold'))
    buttone.place(anchor=CENTER, x=440, y=230)

    sc.protocol("WM_DELETE_WINDOW", exit_sc)

    # 窗体显示
    sc.mainloop()