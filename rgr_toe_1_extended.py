from tkinter import Tk, Label, Entry, Button, Frame, messagebox, END
from tkinter.ttk import Combobox
import matplotlib.pyplot as plt
import numpy as np
import math, re

a = complex(-1/2,math.sqrt(3)/2)
test_data = ['24j', '19j', '9j', '5+5j', '3+4j', '1+3j', '40+30j', '50+40j', '25+5j', '127', '5']

def compute_equivalents(Efg, z, zn):
    Ee1 = Efg*z[1,-1]/(np.sum(z[1]))
    ze1 = z[1,2]*(z[1,0]+z[1,1])/(np.sum(z[1]))
    ze2 = z[2,2]*(z[2,0]+z[2,1])/(np.sum(z[2]))
    ze0 = z[0,2]*(z[0,0]+z[0,1]+3*zn)/(np.sum(z[0])+3*zn)
    return Ee1, ze1, ze2, ze0

def solve_cross(Efg, z, zn, asymmetry_phase):
    global a
    U_phase = np.array([None, None, None])
    I_phase = np.array([None, None, None])
    asymmetry_phase = np.array(asymmetry_phase)

    # Токи и напряжения фаз несимметричной нагрузки для данного вида несимметрии
    U_phase[(asymmetry_phase == 1)[:-1]] = 0
    I_phase[(asymmetry_phase == 0)[:-1]] = 0

    Ee1, ze1, ze2, ze0 = compute_equivalents(Efg, z, zn)
    
    A = None
    B = None
    
    # Если фаза(-ы) закорочена(-ы) на землю
    if asymmetry_phase[-1] == 1:
        tmp_a = [[1,1,1], [0,0,0]]
        tmp_b = [[a**2,a,1], [0,0,0]]
        tmp_c = [[a,a**2,1], [0,0,0]]
        A = [
            [ze1, 0, 0, 1, 0, 0],
            [0, ze2, 0, 0, 1, 0],
            [0, 0, ze0, 0, 0, 1],
            np.append(tmp_a[asymmetry_phase[0]], tmp_a[asymmetry_phase[0]-1]),
            np.append(tmp_b[asymmetry_phase[1]], tmp_b[asymmetry_phase[1]-1]),
            np.append(tmp_c[asymmetry_phase[2]], tmp_c[asymmetry_phase[2]-1])
        ]
        B = [
            Ee1,
            0,
            0,
            0,
            0,
            0
        ]
        I1, I2, I0, U1, U2, U0 = np.linalg.solve(A,B)
    # Если фазы не закорочены на землю
    elif asymmetry_phase[-1] == 0 and asymmetry_phase.sum() == 2:
        I0 = 0
        U0 = 0
        tmp = [np.array([1,1]), np.array([a**2,a]), np.array([a,a**2])]
        phase_filter = (asymmetry_phase==0)[:-1]
        non_cross_phase_ind = phase_filter.tolist().index(True)
        cross_phase_1st_ind = phase_filter.tolist().index(False)
        cross_phase_2nd_ind = phase_filter.tolist()[cross_phase_1st_ind+1:].index(False)+cross_phase_1st_ind+1
        A = [
            [ze1, 0, 1, 0],
            [0, ze2, 0, 1],
            np.append(tmp[non_cross_phase_ind], [0,0]),
            np.append([0,0], tmp[cross_phase_1st_ind]-tmp[cross_phase_2nd_ind])
        ]
        B = [
            Ee1,
            0,
            0,
            0
        ]
        I1, I2, U1, U2 = np.linalg.solve(A,B)
    
    return I1, I2, I0, U1, U2, U0, I_phase, U_phase

def solve_longtitude(Efg, z, zn, asymmetry_phase):
    global a    
    I_phase = np.array([None, None, None])
    U_phase = np.array([None, None, None])
    asymmetry_phase = np.array(asymmetry_phase)
    
    I_phase[asymmetry_phase==1] = 0
    U_phase[asymmetry_phase==0] = 0

    tmp_a = [[1,1,1],[0,0,0]]
    tmp_b = [[a**2,a,1],[0,0,0]]
    tmp_c = [[a,a**2,1],[0,0,0]]
    A = [
        [z[1,0]+z[1,1]+z[1,2],0,0,1,0,0],
        [0,z[2,0]+z[2,1]+z[2,2],0,0,1,0],
        [0,0,z[0,0]+z[0,1]+z[0,2]+3*zn,0,0,1],
        np.append(tmp_a[asymmetry_phase[0]-1], tmp_a[asymmetry_phase[0]]),
        np.append(tmp_b[asymmetry_phase[1]-1], tmp_b[asymmetry_phase[1]]),
        np.append(tmp_c[asymmetry_phase[2]-1], tmp_c[asymmetry_phase[2]])
    ]
    B = [
        Efg,
        0,
        0,
        0,
        0,
        0
    ]
    I1, I2, I0, U1, U2, U0 = np.linalg.solve(A,B)

    return I1, I2, I0, U1, U2, U0, I_phase, U_phase

def solve(Efg, z, zn, asymmetry_cfg):
    coef_phase = [[1,1],[a**2,a],[a,a**2]]
    asymmetry_type, asymmetry_phase = asymmetry_cfg

    if asymmetry_type == 0:
        I1, I2, I0, U1, U2, U0, I_phase, U_phase = solve_cross(Efg, z, zn, asymmetry_phase)
    elif asymmetry_type == 1:
        I1, I2, I0, U1, U2, U0, I_phase, U_phase = solve_longtitude(Efg, z, zn, asymmetry_phase)

    for i in range(3):
        if I_phase[i] != 0:
            I_phase[i] = I1*coef_phase[i][0]+I2*coef_phase[i][1]+I0
        if U_phase[i] != 0:
            U_phase[i] = U1*coef_phase[i][0]+U2*coef_phase[i][1]+U0

    return [I1, I2, I0], [U1, U2, U0], I_phase, U_phase

def complex_to_str(z):
    s = ''
    if round(z.imag,2) < 0:
        s = '{}{}j'.format(round(z.real,2), round(z.imag,2))
    elif round(z.imag,2) == 0: 
        s = '{}'.format(round(z.real,2))
    elif round(z.imag,2) > 0:
        s = '{}+{}j'.format(round(z.real,2), round(z.imag,2))
    return s


def answer_to_latex(s):
    global a
    tex = r'$I_1 = I_{A1}, I_2 = I_{A2}, I_0 = I_{A0}$'+'\n'+r'$U_1 = U_{A1}, U_2 = U_{A2}, U_0 = U_{A0}$'
    I_phases_solve = ['I_1+I_2+I_0', 'a^2*I_1+a*I_2+I_0', 'a*I_1+a^2*I_2+I_0']
    U_phases_solve = ['U_1+U_2+U_0', 'a^2*U_1+a*U_2+U_0', 'a*U_1+a^2*U_2+U_0']
    Phases_coefs = [[1,1,1], [a**2,a,1], [a,a**2,1]]
    phases = ['A', 'B', 'C']
    idxs = [1,2,0]
    I_symmetry, U_symmetry, I_phase, U_phase = s

    for i in range(3):
        tex += '\n$I_{} = {}$'.format(idxs[i], complex_to_str(I_symmetry[i]))

    for i in range(3):
        tex += '\n$U_{} = {}$'.format(idxs[i], complex_to_str(U_symmetry[i]))

    for i in range(3):
        tex += '\n$I_{} = {} = {}$'.format(phases[i], I_phases_solve[i], complex_to_str(I_phase[i]))

    for i in range(3):
        tex += '\n$U_{} = {} = {}$'.format(phases[i], U_phases_solve[i], complex_to_str(U_phase[i]))
    return tex

def show_solve(s):
    tex = answer_to_latex(s)
    fig = plt.figure()
    fig.canvas.set_window_title('Ответ')
    ax = fig.add_axes([0,0,1,1])
    ax.set_axis_off()

    t = ax.text(0.5, 0.5, tex,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=14, color='black')
        
    ax.figure.canvas.draw()
    bbox = t.get_window_extent()
    print(bbox.width, bbox.height)

    fig.set_size_inches(bbox.width/80,bbox.height/80)
    plt.show(block=False)


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.asymmetry_type = 0
        self.asymmetry_type_dict = ['КЗ', 'ОБРЫВ']
        self.asymmetry_phase_close = {
            'КЗ ФАЗЫ А НА ЗЕМЛЮ' : [1,0,0,1],
            'КЗ ФАЗЫ В НА ЗЕМЛЮ': [0,1,0,1],
            'КЗ ФАЗЫ С НА ЗЕМЛЮ': [0,0,1,1],
            'КЗ ФАЗ А И В НА ЗЕМЛЮ': [1,1,0,1],
            'КЗ ФАЗ А И С НА ЗЕМЛЮ': [1,0,1,1],
            'КЗ ФАЗ В И С НА ЗЕМЛЮ': [0,1,1,1],
            'КЗ МЕЖДУ ФАЗАМИ А И В': [1,1,0,0],
            'КЗ МЕЖДУ ФАЗАМИ А И С': [1,0,1,0],
            'КЗ МЕЖДУ ФАЗАМИ В И С': [0,1,1,0],
        }
        self.asymmetry_phase_open = {
            'ОБРЫВ ФАЗЫ А': [1,0,0],
            'ОБРЫВ ФАЗЫ В': [0,1,0],
            'ОБРЫВ ФАЗЫ С': [0,0,1],
            'ОБРЫВ ФАЗ А И В': [1,1,0],
            'ОБРЫВ ФАЗ А И С': [1,0,1],
            'ОБРЫВ ФАЗ В И С': [0,1,1]
        }
        self.Efg = None
        self.Z = None
        self.Zn = None
        self.create_widgets()

    def combobox_handler(self):
        self.asymmetry_type = self.asymmetry_type_dict.index(self.cmbb_asymm_type.get())
        if self.asymmetry_type == 0:
            self.cmbb_asymm_phase.config(values=list(self.asymmetry_phase_close.keys()))
            self.cmbb_asymm_phase.set(list(self.asymmetry_phase_close.keys())[0])
        elif self.asymmetry_type == 1:
            self.cmbb_asymm_phase.config(values=list(self.asymmetry_phase_open.keys()))
            self.cmbb_asymm_phase.set(list(self.asymmetry_phase_open.keys())[0])
        else:
            self.cmbb_asymm_phase.config(values=[])

    def update_asymmetry_config(self):
        self.asymmetry_cfg = [self.asymmetry_type_dict.index(self.cmbb_asymm_type.get())]
        if self.asymmetry_type == 0:
            self.asymmetry_cfg.append(self.asymmetry_phase_close[self.cmbb_asymm_phase.get()])
        elif self.asymmetry_type == 1:
            self.asymmetry_cfg.append(self.asymmetry_phase_open[self.cmbb_asymm_phase.get()])

    def create_widgets(self):
        keys = ['Zг1', 'Zг2', 'Zг0', 'Zл1', 'Zл2', 'Zл0', 'Zн1', 'Zн2', 'Zн0', 'Eфг', 'Zn']
        self.entries = list()
        
        fr = Frame(self)        
        Label(fr, text='Тип несимметрии').grid(row=0,column=0,pady=2)
        self.cmbb_asymm_type = Combobox(fr, state='readonly', values=self.asymmetry_type_dict, width=25)
        self.cmbb_asymm_type.grid(row=0,column=1,pady=2)
        self.cmbb_asymm_type.set(self.asymmetry_type_dict[0])

        Label(fr, text='Вид несимметрии').grid(row=1,column=0,pady=2)
        self.cmbb_asymm_phase = Combobox(fr, state='readonly', width=25, values=list(self.asymmetry_phase_close.keys()))
        self.cmbb_asymm_phase.grid(row=1,column=1,pady=2)
        self.cmbb_asymm_phase.set(list(self.asymmetry_phase_close.keys())[0])
        
        self.cmbb_asymm_type.bind('<<ComboboxSelected>>', lambda e: self.combobox_handler())

        for i in range(len(keys)):
            l = Label(fr, text='Введите {}:'.format(keys[i]))
            l.grid(row=i+2,column=0)
            e = Entry(fr)
            e.grid(row=i+2,column=1)
            self.entries.append(e)
        fr.pack(side='top')
        
        buttons_fr = Frame(self)

        Button(buttons_fr, text='Рассчитать', command=self.solve_bth_handler).grid(row=0,column=2,padx=10,pady=4)
        Button(buttons_fr, text='Test fill', command=self.fill_test).grid(row=0,column=1,padx=10,pady=4)
        Button(buttons_fr, text='Очистить', command=self.clear_entries).grid(row=0,column=0,padx=10,pady=4)

        buttons_fr.pack(side='bottom')

    def fill_test(self):
        for i in range(len(self.entries)):
            self.entries[i].delete(0,END)
            self.entries[i].insert(0,test_data[i])

    def clear_entries(self):
        for e in self.entries:
            e.delete(0,END)

    def msg(self, txt):
        messagebox.showerror('Error', txt)

    def str_to_complex(self, data):
        validate = []
        pattern = r'((-?[0-9]+|([0-9]+\.[0-9]+))[+-](([0-9]*|([0-9]+\.[0-9]+))j))|(-?[0-9]+|([0-9]+\.[0-9]+))|(-?([0-9]*|([0-9]+\.[0-9]+))j)'
        for string in data:
            validate.append(re.fullmatch(pattern, string))
        if np.any((np.array(validate)!=None) == False):
            self.msg('Введены некорректные значения.')
            return None
        for i in range(len(data)):
            data[i] = complex(data[i])
        return data

    def solve_bth_handler(self):
        data = list()
        for entry in self.entries:
            data.append(entry.get())
        data = self.str_to_complex(data)
        print(data)
        if data == None:
            return
        self.Efg = data[-2]
        self.Zn = data[-1]
        self.Z = np.array([[data[2], data[5], data[-3]], [data[0], data[3], data[6]], [data[1], data[4], data[7]]])
        self.update_asymmetry_config()
        final_solve = solve(self.Efg, self.Z, self.Zn, self.asymmetry_cfg)
        # print(final_solve)
        try:
            show_solve(final_solve)
        except Exception as e:
            print(e)
            exit(1)

root = Tk()
root.minsize(height=300, width=280)
root.resizable(0,0)
app = Application(master=root)
app.mainloop()