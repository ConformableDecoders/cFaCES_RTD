import tkinter as tk
from tkinter import * 

def display_prediction(labels, weighted_probs):
	out = ''
	for motion in weighted_probs[0]:
		out += labels[motion[0]] + '    '+ str(round(100*motion[1], 1)) + '%'
		out += '\n'
	# print (out)
	root = Tk()
	root.geometry('600x600+300+300')
	root.wm_title('Motion Prediction')
	msg = Message(root, text=out)
	msg.config(font=('consolas', 40))
	msg.pack()
	root.mainloop()