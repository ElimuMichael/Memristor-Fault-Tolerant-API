import subprocess
import PySimpleGUI as sg 

def ExecuteCommandSubprocess(command, *args):
	try:
		sp = subprocess.Popen([command, *args], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		out, err = sp.communicate()
		if out:
			sg.Print(out.decode("utf-8"))
		if err:
			sg.Print(err.decode("utf-8"))
		except:
			pass  