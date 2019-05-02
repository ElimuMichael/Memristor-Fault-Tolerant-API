import PySimpleGUI as sg
import time
import sys

sg.ChangeLookAndFeel('Dark')

# ------ Menu Definition ------ #
menu_def = [['File', ['Open', 'Save', 'Exit', 'Properties', 'Install Dependencies']],
            ['Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
            ['Help', 'About'], ]

layout = [
    [sg.Menu(menu_def)],
    [sg.Text('Memristor Model - Hunan University', size=(
        30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
    [sg.Frame(layout=[
        [sg.Text('Resistance On', size=(15, 1), auto_size_text=False,
                 justification='right'), sg.InputText('', key="_ron_")],
        [sg.Text('Resistance Off', size=(15, 1), auto_size_text=False,
                 justification='right'), sg.InputText('', key='_roff_')],
        [sg.Text('Logic Levels', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(
            ('2', '4', '8', '16', '32'), size=(20, 1), key='_logic_')]], title='Memristor Parameters', title_color='red', relief=sg.RELIEF_RIDGE)],
    [sg.Frame(layout=[
        [sg.Text('Retrain', size=(15, 1), auto_size_text=False, justification='right'), sg.Checkbox('Retrain', size=(10, 1), key='_retrain_')], [sg.Text('Dataset', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(
            ('MNIST', 'FashionMNIST', 'Cifar10', 'Cifar100'), size=(20, 1), key='_data_')]], title='Data Option', title_color='red', relief=sg.RELIEF_RIDGE)],

    [sg.Frame(layout=[
        [sg.Text('Percentage', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(
            ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'), size=(20, 1), key='_fault_')],
        [sg.Text('Distribution', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(
            ('Entire Crossbar', 'Per Split'), size=(20, 1), key='_dist_')],
        [sg.Text('Seed Value', size=(15, 1), auto_size_text=False, justification='right'), sg.InputText('', key='_seed_')]], title='Fault Distribution', title_color='red', relief=sg.RELIEF_RIDGE)],

    [sg.Frame(layout=[
        [sg.Text('Tolerance Method', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(
            ('None', 'Existing Cells', 'Redundant Columns', 'Combined', 'Proposed Split'), size=(20, 1), key='_recovery_')],
        [sg.Text('Crossbar Split', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(('4', '8'), size=(20, 1), key='_split_')]], title='Fault Tolerance', title_color='red', relief=sg.RELIEF_RIDGE)],
    [sg.Text('_' * 80)],
    [sg.Text('External Files', size=(35, 1))],
    [sg.Text('User Model', size=(15, 1), auto_size_text=False, justification='right'),
        sg.InputCombo(('Lenet5', 'AlexNet', 'GoogleNet'), size=(20, 1), key='_model_')],
    # [sg.Text('Saved JSON Model', size=(15, 1), auto_size_text=False, justification='right'),
    #     sg.InputText(''), sg.FileBrowse()],
    [sg.Text('Pretrained Weights', size=(15, 1), auto_size_text=False, justification='right'),
        sg.InputText('', key='_weight_'), sg.FileBrowse(file_types=(("ALL Files", "*.npy"),))],
    [sg.Submit(), sg.Button('Exit', button_color=('white','red'))]
]


window = sg.Window('Memristor Fault Analysis', default_element_size=(
    40, 1), grab_anywhere=False, icon='icon/hunan_university_icon.ico').Layout(layout)


# Default Result Windows = False
result_window = False
error_window = False
req_window = False
current_time = 0
paused = False
# Keep the window Open
while True:
    event, values = window.Read()

    if event is None or event == 'Exit':
        break

    elif event == 'About':
        window.Disappear()
        sg.Popup('This is a memristor Fault Simulation Model',
                 'Designed Using Python',
                 'For any information or improvements, contact us on elimumichael@hnu.edu.cn',
                 'Copyright: Hunan University')
        window.Reappear()

    elif event == 'Properties':
        pass

    elif event == 'Install Dependencies':
        req_window = True
        result_window = False
        layout_req = [
            [sg.Text('Click the Check requirements button to identify the missing packages', key='_req_')], [sg.Button('Check'), sg.Button('Exit', button_color=('white', 'red'))]]
        req = sg.Window('Required Packages', icon='icon/hunan_university_icon.ico').Layout(layout_req)

        while True:
            window.Hide()
            event_req, value_req = req.Read()
            if event_req is None or event_req == 'Exit':
                req_window = False
                req.Close()
                break
            elif event_req == 'Check':
                import packageVerifier
                pkg = packageVerifier.check_packages()
                req.FindElement('_req_').Update(pkg)
        window.UnHide()

    elif event == 'Open':
        filename = sg.PopupGetFile('File To Get', no_window=True)

    elif not result_window and not error_window and event == 'Submit':

        result_window = True
        # Verufy that all the Values have been filled
        output_values = []
        for k, val in values.items():
            output_values.append(val)
        if '' in output_values:
            error_window = True
            result_window = False
            layout_err = [
                [sg.Text('Please Enter all the required Parameters'), sg.Button('Exit', button_color=('white', 'red'))]]
            err = sg.Window('Error', icon='icon/hunan_university_icon.ico').Layout(layout_err)

            while True:
                window.Hide()
                event_err, value_err = err.Read(timeout=0)
                if event_err is None or event_err == 'Exit':
                    error_window = False
                    err.Close()
                    break
            window.UnHide()

        else:
            window.Hide()

            # Inputs and adjustments
            col = [[sg.Frame(layout=[
                    [sg.Text('On Resistance: ', size=(15, 1), auto_size_text=False, justification='right'), sg.Text(values['_ron_'])],
                    [sg.Text('Off Resistance: ', size=(15, 1), auto_size_text=False, justification='right'), sg.Text(values['_roff_'])],
                    [sg.Text('Logic Levels: ', size=(15, 1), auto_size_text=False, justification='right'), sg.Text(values['_logic_'])],
                    [sg.Text('Retrain: ', size=(15, 1), auto_size_text=False, justification='right'), sg.Text(values['_retrain_'])],
                    [sg.Text('Seed Value: ', size=(15, 1), auto_size_text=False, justification='right'), sg.Text(values['_seed_'])],
                    [sg.Text('Distribution', size=(15, 1), auto_size_text=False, justification='right'), sg.Text(values['_dist_'])],
                    [sg.Text('Fault Percentage: ', size=(15, 1), auto_size_text=False, justification='right'), sg.Text(values['_fault_'])],
                    [sg.Text('Recovery Method: ', size=(15, 1), auto_size_text=False, justification='right'), sg.Text(values['_recovery_'])]], title='Parameters - ' + values['_model_'], title_color='red', relief=sg.RELIEF_SUNKEN)], 

                # Parameter Adjustments
                [sg.Frame(layout=[
                    [sg.Text('Distribution', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(('Entire Crossbar', 'Per Split'), size=(20, 1), key='_dist_adj_', default_value=values['_dist_'])],
                    [sg.Text('Logic Levels: ', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(('2', '4', '8', '16', '32'), size=(20, 1), key='_logic_adj_', default_value=values['_logic_'])],
                    [sg.Text('Retrain: ', size=(15, 1), auto_size_text=False, justification='right'), sg.Checkbox('Retrain', size=(10, 1), key='_retrain_adj_', default=values['_retrain_'])],
                    [sg.Text('Seed Value: ', size=(15, 1), auto_size_text=False, justification='right'),  sg.InputText(default_text=values['_seed_'], key='_seed_adj_', size=(15, 1))],
                    [sg.Text('Fault Percentage: ', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'), size=(20, 1), key='_fault_adj_', default_value=values['_fault_'])],
                    [sg.Text('Recovery Method: ', size=(15, 1), auto_size_text=False, justification='right'), sg.InputCombo(('None', 'Existing Cells', 'Redundant Columns', 'Combined', 'Proposed Split'), size=(20, 1), key='_recovery_adj_', default_value=values['_recovery_'])]], title='Parameters Adjustment', title_color='red', relief=sg.RELIEF_SUNKEN)]]



            # First Tab Information
            tab1_layout = [
                [sg.Text('Seed: ', size=(15, 1), auto_size_text=False, justification='left'), sg.Text(values['_seed_'])],
                [sg.Text('Weight File: ', size=(15, 1), auto_size_text=False, justification='left'), sg.Text(values['_weight_'])],
                [sg.ProgressBar(10000, orientation='h', size=(65, 20), key='progressbar')],
                [sg.Button('Run Model')],
                [sg.Text('Software Based Accuracy: ', size=(15, 1), auto_size_text=False, justification='left'), sg.InputText(disabled=True, text_color='#000000', key='_acc_soft_'), sg.Text(' (%)', justification='left')],
                [sg.Text('Memristor Based Accuracy: ', size=(15, 1), auto_size_text=False, justification='left'), sg.InputText(disabled=True, text_color='#000000', key='_acc_'), sg.Text(' (%)', justification='left')],
                [sg.Text('Retrained Accuracy: ', size=(15, 1), auto_size_text=False, justification='left'), sg.InputText(disabled=True, text_color='#000000', key='_accretrained_'), sg.Text(' (%)', justification='left')],
                [sg.Output(size=(100, 16), key='output')], [sg.Text('History')], [sg.Output(size=(100, 8), key='_history_'), sg.Button('Clear', key='_clear_')]]
            

            # Second Tab Information
            tab2_layout = [
                [sg.Frame(layout=[
                    [sg.Text('Image', size=(15, 1), auto_size_text=False, justification='left'),
                    sg.InputText('', key='_pred_img_'), sg.FileBrowse()], 
                    [sg.Button('Predict')],
                    [sg.Text('Prediction', size=(10, 1), auto_size_text=False, justification='left'),
                    sg.InputText(disabled=True, key='_netPred_'), sg.Text('Accuracy', size=(10, 1), auto_size_text=False, justification='left'),
                    sg.InputText(disabled=True, key='_acc_')]], title='Test Model', title_color='red', relief=sg.RELIEF_RIDGE)],
                    
                    [sg.Frame(layout=[
            [sg.Text('Layer', size=(15, 1), auto_size_text=False, justification='left'), sg.InputCombo(('Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5' ), size=(20, 1), key='_layer_')], [sg.Text('Component', size=(15, 1), auto_size_text=False, justification='left'), sg.InputCombo(('Fault Distribution', 'Convolution Output'), size=(20, 1), key='_visual_')], [sg.Button('View', button_color=('white', 'orange'), size=(10, 1))]], title='Prediction And Fault Visualization', title_color='red', relief=sg.RELIEF_RIDGE)],
            [sg.Image(filename='./Test Image/download.png', data=None, size=(100, 8), key='_visualization_', visible=False)]
        ]
            
            # Third Tab Information
            tab3_layout = [[sg.T('Network Log')], [sg.Output(size=(100, 18), key='output3')], [sg.T('Model Summary'), sg.Button('Model Summary')], [sg.Output(size=(100, 18), key='_mod_summ_')]]
            
            # Apply To the Lyout
            layout2 = [
                # Input Feeds
                sg.Column(col),
                # Tabbed groups
                sg.TabGroup([[sg.Tab('Model Prediction', tab1_layout), sg.Tab(
                    'Visualization', tab2_layout), sg.Tab('Network Information', tab3_layout)]])], [sg.RButton('Exit')]

            result = sg.Window('Results',icon='icon/hunan_university_icon.ico').Layout(layout2)

            while True:
                event2, values2 = result.Read()

                if event2 is None or event2 == 'Exit':
                    result_window = False
                    result.Close()
                    window.UnHide()
                    break
                
                # Network Information
                
                Ron = int(values['_ron_'])
                Roff = int(values['_roff_'])
                logic_levels = int(values['_logic_'])
                weightURL = values['_weight_']
                test_model = values['_model_']
                split = int(values['_split_'])
                case_considered = values['_dist_']
                data = values['_data_']

                # Obtain the genuine Test case for the memristor
                test_case = 'crossbar_no_split'
                if case_considered == 'Per Split':
                    test_case = 'crossbar_with_split'

                # Get the seed value used to maintain consistency
                seed_val = int(values['_seed_'])
                recovery = values['_recovery_']
                retrain = values['_retrain_']
                faults = int(values['_fault_'])

                # Define the kind of approach handled
                if recovery == 'Proposed Split':
                    approach = 'proposed'
                else:
                    approach = 'normal'

                recovery = recovery.replace(' ', '_').lower()
                print(recovery)

                # Enter Parametrs to the memristor model
                import Weight_converter_final as mem
                import json
                import numpy as np
                import test
                from util import get_data
                from PIL import Image
                # Get the Memristance values of the meristor
                cond_vals = mem.conductanceValues(Ron, Roff)
                
                
                # Split of Conductance Values
                MINcond = cond_vals['c_min']
                MAXcond = cond_vals['c_max']
                condRANGE = cond_vals['c_range']
                
                # # Considering the bit levels
                num_bits = logic_levels
                
                # # model = eval(input('Enter model (lenet5, FashionMnist, ..): '))
                
                model = test.Lenet5()
                if test_model == 'Lenet5':
                    model = model
                elif test_model == 'Alexnet':
                    pass
                elif test_model == 'GoogleNet':
                    pass
                
                # # Get the dataset
                dataset = get_data(data)
                X_train = dataset['x_train']
                y_train_cat = dataset['y_train']
                X_test = dataset['x_test']
                y_test_cat = dataset['y_test']
                batch_size = dataset['batch_size']
                num_classes = dataset['num_classes']
                input_shape = dataset['input_shape']

                # print(model.summary(), dataset)
                
                default_weights = weightURL

                if event2 == 'Run Model':
                
                    # default_weights = 'fashion_mnist_weight.npy'
                    # default_weights = 'alexnet_weights.npy'
                    # default_weights = 'inception_cifar10.npy'
                    # print(X_train, y_train_cat, X_test, y_test_cat)
                    
                    # print(default_weights)

                    # # Software Based Accuracy
                    if split == 8:
                        results.Hide()
                        sg.Popup('Sorry! The use of {} has not yet been implemented. Please try using 4 split factor'.format(split))
                        break;

                    elif  test_model == 'AlexNet' or test_model == 'GoogleNet':
                        result.Hide()
                        sg.Popup('Sorry!, {} is still under development. Try using a Lenet Network model'.format(test_model))
                        break;

                    elif  data == "FashionMNIST" or data == "Cifar10" or data == "Cifar100":
                        result.Hide()
                        sg.Popup('Sorry!, {} is still under development. Please try using a network trained on MNIST dataset'.format(data))
                        break;

                    
                    # results.unHide()
                    weights_def = np.load(default_weights)
                    model.set_weights(weights_def)
                    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

                    # Test Model Accuracy
                    Accuracy_def = model.evaluate(X_test, y_test_cat)

                    # # Memristor Trained Network
                    converted_weights = mem.convert_weight(cond_vals, num_bits, default_weights, faults, split, approach, recovery, test_case, seed_val=seed_val)

                    weight_checker, weight_file = mem.accuracy_check(converted_weights, new_weight_filename='memristorWeights/Memristor_weights.npy')

                    # #Model Accuracies
                    weight_file = 'memristorWeights/Memristor_weights.npy'
                    weights = np.load(weight_file)
                    model.set_weights(weights)
                    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

                    # Test Model Accuracy
                    Accuracy_test = model.evaluate(X_test, y_test_cat)
                    Accuracy_retrained = Accuracy_test
                    
                    
                    sg.Print(Accuracy_def, Accuracy_test)

                    
                    result.FindElement('_acc_soft_').Update(Accuracy_def[1])
                    result.FindElement('_acc_').Update(Accuracy_test[1])
                    result.FindElement('_accretrained_').Update(Accuracy_retrained[1])
                    # if retrain ==True:
                    #     model.fit(X_train, y_train_cat, batch_size=128, validation_split=0.2)
        
                    #     x_weights = weight_checker['x_weights']
                    #     final_weight = weight_checker['weight']
                    #     weight_faultfree = weight_checker['weight_faultfree']
                        
                    #     retrained_weights = np.array(model.get_weights())
                    #     model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
                    #     Accuracy_retrained_no_fault = model.evaluate(X_test, y_test_cat)
                    #     # print('Retrained Without Fault Consideration: {}**'.format(Accuracy_retrained_no_fault[1]))
                        
                    #     for i in range(x_weights.shape[0]):
                    #         my_shape = x_weights[i].shape
                    #         # print(my_shape)
                    #         if len(my_shape) == 1:
                    #             for j in range(x_weights[i].shape[0]):
                    #                 if x_weights[i][j] == False:
                    #                     retrained_weights[i][j] = final_weight[i][j]
                                        
                    #         elif len(my_shape) == 2:
                    #             for j in range(my_shape[0]):
                    #                 for k in range(my_shape[1]):                        
                    #                     if x_weights[i][j][k] == False:
                    #                         retrained_weights[i][j][k] = final_weight[i][j][k]
                    #         elif len(my_shape) == 4:
                    #             for j in range(my_shape[0]):
                    #                 for k in range(my_shape[1]): 
                    #                     for m in range(my_shape[2]):
                    #                         for n in range(my_shape[3]): 
                    #                             if x_weights[i][j][k][m][n] == False:
                    #                                 retrained_weights[i][j][k][m][n] = final_weight[i][j][k][m][n]
                                            
                    #     model.set_weights(retrained_weights)
                    #     model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
                    #     Accuracy_retrained = model.evaluate(X_test, y_test_cat)

                    
                    # sg.Print(Accuracy_def)
                    
                    
                
                
                # Prediction
                if event2 == 'Predict':
                    # sg.Print(event2)
                    # import matplotlib.pyplot as plt
                    # img = Image.open(values2['_pred_img_'])
                    # import cv2
                    # img_gray = cv2.cv2.cvtColor(img, cv2.cv2.COLOR_RGB2GRAY)
                    # plt.imshow(img_gray)
                    # plt.show()
                    pass
                    # result.FindElement('_visualization_').Update(plt.show())
                
                # Visualization
                if event2 == 'View':
                    img_filename = values2['_layer_']+'_'+values['_fault_']+'_'+values['_seed_']+'_'+values['_recovery_']+'_'+values2['_visual_']+'.png'
                    img = img_filename.replace(' ', '_').lower()
                    my_img = Image.open(img)

                    result.FindElement('_visualization_').Update(data=my_img, visible=True)

                if event2 == 'Model Summary':                    
                    result.FindElement('_mod_summ_').Update(model.summary())
                

            # print(event2, values2)
                    
                    
                    # model = util.model
                    # converted_weights = np.array([])
                    # software_based_accuracy = model(converted_weights, default_weights, 'Ideal')

                    # print(f'Ideal/Software Based Accuracy: {software_based_accuracy[1]}\n{"+"*40}')
                    
                    # # Define Approach to Use
                    # approach = case_considered
                    
                    # # Define Fix Model to Use
                    
                    # filter_size=split
                    # fix_model = recovery
                    
                    # # Introduce Fault Defects
                    # # percentage_defect = int(input("Enter Fault Percentage: "))
                    # with open ('accuracy_tracker.txt', 'a+') as f:
                    #     f.write("{} - {} - {} bits\n".format(fix_model, test_case,num_bits))
                    #     f.write("perc_def\tAccuracyies\tAverage\n")
                    #     f.write("_"*40+"\n")
                    
                    # # seed_vals = [123, 111, 155, 555, 100,789, 329, 500, 907, 644]
                        
                    # seed_vals = seed_val
                    
                    # for percentage_defect in range(faults, faults+1):
                        
                    #     accs = []
                    #     acc_retrain= []
                    #     re_train = retrain

                    #     converted_weights = mem.convert_weight(cond_vals, num_bits, default_weights, percentage_defect, filter_size, approach, fix_model, test_case, seed_val=seed_vals)
                    #     # print(converted_weights)
                    #     acc = mem.accuracy_check(converted_weights, percentage_defect, model, new_weight_filename='memristorWeights/Memristor_weights.npy', re_train = re_train)
                    #     accs.append(acc[0])
                    #     acc_retrain.append(acc[1])

                    #     with open ('accuracy_tracker.txt', 'a+') as f:
                    #         f.write("{}\t{}\t{}\n".format(percentage_defect,accs, np.array(accs).mean()))
                    #         f.write("{}\t{}\t{}\n".format(percentage_defect,acc_retrain, np.array(acc_retrain).mean()))

                    #     sg.Print(acc)




                    # output_data = {'dataset': data, 'Fault Tolerance': recovery, 'Percentage': faults, 'seed_val': seed_val, 'test_case': test_case, 'acc':out['acc'], 'retrain':out['retrain']}

                    # with open('history/history.csv', 'w+') as f:
                    #     all_data = f.read()
                    #     print(all_data)
                    #     # all_data['history'].append(output_values)
                    #     # json.dump(all_data, f)

                    # from Weight_converter_new import main
                    # import testModels
                    # import util

                    # result.FindElement('output').Update()

                    # # Output History
                    # hist = util.read_hist()
                    
                    # result.FindElement('_history_').Update(hist)

                    # result.FindElement('_acc_').Update(out)
                    # result.FindElement('_accretrained_').Update(out)
                    # sg.Print(out)

window.Close()
