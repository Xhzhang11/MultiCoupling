"""
This file contains functions that test the behavior of the model
These functions generally involve some psychometric measurements of the model,
for example performance in decision-making tasks as a function of input strength

These measurements are important as they show whether the network exhibits
some critically important computations, including integration and generalization.
"""


from __future__ import division

import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
from matplotlib import pyplot as plt
import run
import tools
import pdb
from matplotlib import cm
import seaborn as sns
from sklearn.decomposition import PCA
c_perf = sns.color_palette("hls", 8)#muted
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def get_epoch(model_dir,rule_name,hp):

    runnerObj = run.Runner(rule_name=rule_name, hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='test')
    trial_input, run_result = runnerObj.run(batch_size=1, gaussian_center1=1, gaussian_center2=0.5,cue_sign=1)
    stim1_on, stim1_off = trial_input.epochs['stim1']
    stim2_on, stim2_off = trial_input.epochs['stim2']
    response_on, response_off = trial_input.epochs['response']


    epoch = {'stim2_on':stim2_on,
             'stim2_off':stim2_off,
             'stim1_on':stim1_on,
             'stim1_off':stim1_off,
             'response_on':response_on,
             'response_off':response_off}
    #print('epoch',epoch)

    return epoch



def generate_test_trial(model_dir,hp,rule_name,batch_size=1,
                        gaussian_center1=0,
                        gaussian_center2=0,
                        cue_sign=None):


    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))


    runnerObj = run.Runner(rule_name=rule_name, hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='test')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            gaussian_center1=gaussian_center1,
                                            gaussian_center2=gaussian_center2,
                                            cue_sign=cue_sign)

    return trial_input, run_result


def get_neurons_activity_mean(model_dir,hp,batch_size=1,
                                rule_name='both',
                                gaussian_center1=0,
                                gaussian_center2=0,
                                cue_sign=None):

    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))

    runnerObj = run.Runner(rule_name=rule_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test')
    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            gaussian_center1=gaussian_center1,
                                            gaussian_center2=gaussian_center2,
                                            cue_sign=cue_sign)

    #### average value over batch_sizes for hidden state
    fr_pfc = run_result.firing_rate_binder_pfc.detach().cpu().numpy()
    fr_pfc_list = list([fr_pfc[:, i, :] for i in range(batch_size)])
    fr_pfc_mean = np.mean(np.array(fr_pfc_list), axis=0)

    fr_parietal = run_result.firing_rate_binder_parietal.detach().cpu().numpy()
    fr_parietal_list = list([fr_parietal[:, i, :] for i in range(batch_size)])
    fr_parietal_mean = np.mean(np.array(fr_parietal_list), axis=0)




    return fr_pfc_mean, fr_parietal_mean



def get_allangle_activity_mean(model_dir,hp,batch_size=1,
                                rule_name='both',
                                cue_sign=None):


    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))


    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences
    fr_pfc_angles = []
    fr_parietal_angles = []
    for angle in np.array(pref):

        gaussian_center1 = angle
        gaussian_center2 = (gaussian_center1 + 1 * np.pi)% (2 * np.pi)

        runnerObj = run.Runner(rule_name=rule_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test')
        trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                                gaussian_center1=gaussian_center1,
                                                gaussian_center2=gaussian_center2,
                                                cue_sign=cue_sign)
        #### average value over batch_sizes for hidden state
        fr_pfc = run_result.firing_rate_binder_pfc.detach().cpu().numpy()
        fr_pfc_list = list([fr_pfc[:, i, :] for i in range(batch_size)])
        fr_pfc_mean = np.mean(np.array(fr_pfc_list), axis=0)
        fr_pfc_angles.append(fr_pfc_mean)

        fr_parietal = run_result.firing_rate_binder_parietal.detach().cpu().numpy()
        fr_parietal_list = list([fr_parietal[:, i, :] for i in range(batch_size)])
        fr_parietal_mean = np.mean(np.array(fr_parietal_list), axis=0)
        fr_parietal_angles.append(fr_parietal_mean)


    fr_pfc_angles_array = np.array(fr_pfc_angles)
    fr_pfc_angles_mean=np.mean(np.array(fr_pfc_angles_array), axis=0)

    fr_parietal_angles_array = np.array(fr_parietal_angles)
    fr_parietal_angles_mean = np.mean(np.array(fr_parietal_angles_array), axis=0)

    return fr_parietal_angles_mean, fr_pfc_angles_mean

def get_single_activity(model_dir,hp,batch_size=1,
                                rule_name='both',
                                gaussian_center1=0,
                                gaussian_center2=0,
                                cue_sign=None):

    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))

    runnerObj = run.Runner(rule_name=rule_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test')
    trial_input, run_result = runnerObj.run(batch_size=512,
                                            gaussian_center1=gaussian_center1,
                                            gaussian_center2=gaussian_center2,
                                            cue_sign=cue_sign)

    #### average value over batch_sizes for hidden state
    trial_idx=0
    fr_parietal = run_result.firing_rate_binder_parietal.detach().cpu().numpy()
    fr_parietal_list = list([fr_parietal[:, i, :] for i in range(trial_idx,trial_idx+512)])
    fr_parietal_mean = np.mean(np.array(fr_parietal_list), axis=0)


    fr_pfc = run_result.firing_rate_binder_pfc.detach().cpu().numpy()
    fr_pfc_list = list([fr_pfc[:, i, :] for i in range(trial_idx,trial_idx+512)])
    fr_pfc_mean = np.mean(np.array(fr_pfc_list), axis=0)

    return fr_parietal_mean, fr_pfc_mean



def single_neuron_activity_0(fig_path,model_name, model_dir, idx, hp):

    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    print('epoch',epoch)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]

    fr_parietal_ups=[]
    fr_pfc_ups=[]
    fr_parietal_downs = []
    fr_pfc_downs = []

    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_eachring'])  # preferences
    gaussian_center1 = 0#np.random.choice(pref)
    gaussian_center2 = gaussian_center1 + 1 * np.pi

    exc = int(1*int(hp['n_rnn'] / 2))

    n_trial=300

    for i in range(n_trial):
        fr_parietal_up, fr_pfc_up = get_single_activity(model_dir, hp, batch_size=1, rule_name='retro',
                                                        gaussian_center1=gaussian_center1,
                                                        gaussian_center2=gaussian_center2,
                                                        cue_sign=None)


        # fr_parietal_up, fr_pfc_up = get_allangle_activity_mean(model_dir, hp, batch_size=1, rule_name='prosp',
        #                                                 cue_sign=None)

        # print(fr_pfc_up.shape)

        #print(np.argsort(np.mean(fr_parietal_up,axis=0)),np.mean(fr_parietal_up,axis=0).shape)

        idxs_sort_ppc = np.argsort(np.mean(fr_parietal_up[:,:],axis=0))
        idxs_sort_pfc = np.argsort(np.mean(fr_pfc_up[:,:], axis=0))
        select_idx_ppc = idxs_sort_ppc[150:199]
        select_idx_pfc = idxs_sort_pfc[150:199]

        # fig, axs = plt.subplots(1, 2, figsize=(7.5, 3))
        # axs[0].plot(fr_parietal_up[:, :], color='r', label='up')
        # axs[1].plot(fr_pfc_up[:, :], color='g', label='down')
        # plt.show()
        fr_parietal_ups.append(fr_parietal_up[:, select_idx_ppc])
        fr_pfc_ups.append(fr_pfc_up[:, select_idx_pfc])


    np.save(fig_path + 'fr_parietal_ups.npy', fr_parietal_ups)
    np.save(fig_path + 'fr_pfc_ups.npy', fr_pfc_ups)
    fr_parietal_ups = np.load(fig_path + 'fr_parietal_ups.npy')
    fr_pfc_ups = np.load(fig_path + 'fr_pfc_ups.npy')
    print('fr_parietal_ups',fr_parietal_ups.shape)

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

    accuracy_ppcs = []
    accuracy_pfcs = []
    for time_point in range(stim2_on,stim2_off+10):

        #print(fr_parietal_ups.shape)
        encode_data_pfc = fr_parietal_ups[:,time_point,:]

        labels = np.random.choice([-1,1], size=n_trial)
        labels = np.random.randint(2, size=n_trial)  # Binary labels for simplicity

        #print('labels',labels)

        X_train, X_test, y_train, y_test = train_test_split(encode_data_pfc, labels, test_size=0.2, random_state=42)

        # Train a Support Vector Machine classifier
        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = svm_classifier.predict(X_test)

        # Evaluate the accuracy of the classifier
        accuracy_ppc = accuracy_score(y_test, predictions)
        accuracy_ppcs.append(accuracy_ppc)


    for time_point in range(stim2_on,stim2_off+10):

        #print(fr_parietal_ups.shape)
        encode_data_pfc = fr_pfc_ups[:,time_point,:]
        #print(encode_data_pfc.shape)
        labels = np.random.choice([-1,1], size=n_trial)
        labels = np.random.randint(2, size=n_trial)


        X_train, X_test, y_train, y_test = train_test_split(encode_data_pfc, labels, test_size=0.2, random_state=2)

        # Train a simple Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=2)
        classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = classifier.predict(X_test)

        # Evaluate the accuracy of the classifier
        accuracy_pfc = accuracy_score(y_test, predictions)
        accuracy_pfcs.append(accuracy_pfc)

    print("ppcs", accuracy_ppcs)
    print("pfcs",accuracy_pfcs)

    plt.plot(accuracy_ppcs,c='r',label='ppc')
    plt.plot(accuracy_pfcs, c='g',label='pfc')
    plt.xlabel('from Cue_on',fontsize=15)
    plt.ylabel('classifier accuracy',fontsize=15)
    plt.legend()
    plt.show()


def decode_data(data,target,t):
    signs = target
    fr = data
    time_point = t

    print('signs',signs)

    encode_data = fr[:, time_point, :]

    labels = np.array(signs)
    # labels = np.random.randint(2, size=n_trial)  # Binary labels for simplicity

    X_train, X_test, y_train, y_test = train_test_split(encode_data, labels, test_size=0.2, random_state=42)

    # Train a Support Vector Machine classifier
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = svm_classifier.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy_ppc = accuracy_score(y_test, predictions)


    return accuracy_ppc


def decode_data1(data, target, t, n_iter=5):
    labels = (np.array(target) + 1) // 2  # Convert -1/1 to 0/1
    fr = data[:, t, :]
    accuracies = []

    for _ in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(fr, labels, test_size=0.2)

        # Skip iteration if only one class is present in training
        if len(np.unique(y_train)) < 2:
            continue

        clf = SVC(kernel='linear', C=1.0)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, pred))


    print(accuracies)

    if len(accuracies) == 0:
        return np.nan  # or 0.5 as chance level

    return np.mean(accuracies)

def decode_up_low_both2(fig_path,model_name, model_dir, idx, hp,sigma_x=0.01):
    hp['sigma_x'] = sigma_x

    figure_path = os.path.join(fig_path, 'decode_up_low_both'+'/')
    tools.mkdir_p(figure_path)
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    print('epoch',epoch)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]


    select_cell_ppc = np.arange(0,200)#np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))
    select_cell_pfc = np.arange(0,200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))

    n_trial=300

    fr_parietals = []
    fr_pfcs = []
    labels = []

    #pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_eachring'])  # preferences
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences

    stim_dist = np.random.choice([1 * np.pi / 2, -1 * np.pi / 2],size=n_trial)
    print('pref',pref)
    print('stim_dist', stim_dist)
    random_pref = np.random.choice(pref, size=n_trial, replace=True)
    random_sign = np.random.choice([-1, 1],size=n_trial)


    for i in range(n_trial):
        print()

        gaussian_center1 = random_pref[6]
        gaussian_center2 = (gaussian_center1 + stim_dist[0]) % (2 * np.pi)


        print('==gaussian_center1',gaussian_center1)
        print('gaussian_center2', gaussian_center2)

        sign = random_sign[i]

        fr_parietal, fr_pfc_up = get_single_activity(model_dir, hp, batch_size=512, rule_name='prosp',
                                                        gaussian_center1=gaussian_center1,
                                                        gaussian_center2=gaussian_center2,
                                                        cue_sign=1)



        fr_parietal, fr_pfc_down = get_single_activity(model_dir, hp, batch_size=512, rule_name='prosp',
                                                  gaussian_center1=gaussian_center1,
                                                  gaussian_center2=gaussian_center2,
                                                  cue_sign=-1)



        threshold =0.5
        mean_up = np.where(np.mean(fr_pfc_up[0:stim2_off, :], axis=0) > threshold)[0]
        print('==mean_up', mean_up)

        mean_down = np.where(np.mean(fr_pfc_down[0:stim2_off, :], axis=0) > threshold)[0]
        print('==mean_down', mean_down)

        fig, axs = plt.subplots(1, 2, figsize=(7, 3.2),sharey=True)
        fig.subplots_adjust(top=0.85, bottom=0.1, right=0.95, left=0.06, hspace=0.2, wspace=0.3)
        axs[0].plot(fr_pfc_up[2:response_off, :], label='up')
        axs[1].plot(fr_pfc_down[2:response_off, :], label='down')


        plt.ylim([0,3])

        plt.show()




        sys.exit(0)

        # print('fr_pfc',fr_pfc.shape)





        labels.append(sign)
        fr_parietals.append(fr_parietal[:, select_cell_ppc])
        fr_pfcs.append(fr_pfc[:, select_cell_pfc])





    fr_parietals=np.array(fr_parietals)
    fr_pfcs = np.array(fr_pfcs)
    labels = np.array(labels)

    print('fr_parietals=',fr_parietals.shape)
    print('labels=',labels.shape)




    start = stim1_off
    accuracy_ppcs = []
    accuracy_pfcs = []
    for time_point in range(start,stim2_off):
        accuracy_ppc = decode_data(data=fr_parietals,target=labels,t=time_point)
        accuracy_ppcs.append(accuracy_ppc)

    for time_point in range(start,stim2_off):
        accuracy_pfc = decode_data(data=fr_pfcs, target=labels, t=time_point)
        accuracy_pfcs.append(accuracy_pfc)

    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.15, hspace=0.2, wspace=0.3)
    fig.suptitle(model_name + '/' + str(idx), fontsize=9)

    # print("ppcs", accuracy_ppcs)
    # print("pfcs",accuracy_pfcs)
    plt.plot(accuracy_ppcs,c='g',label='ppc')
    plt.plot(accuracy_pfcs,'-o', c='r',label='pfc')
    axs.axvline(stim1_off-start, color='darkgrey', linestyle='--',label='stim_on')
    axs.axvline(stim2_on-1-start, color='darkgrey', linestyle='-', label='cue_on')
    axs.axvspan(stim1_off-start, stim2_off-start, color='0.95')
    plt.xlabel('from Cue_on',fontsize=15)
    plt.ylabel('classifier accuracy',fontsize=15)
    plt.legend()
    plt.savefig(figure_path + model_name+'_'+str(idx) + 'Select.png')
    plt.show()
def decode_up_low_both1(fig_path,model_name, model_dir, idx, hp,sigma_x=0.01):
    hp['sigma_x'] = sigma_x

    figure_path = os.path.join(fig_path, 'decode_up_low_both'+'/')
    tools.mkdir_p(figure_path)
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    print('epoch',epoch)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]


    select_cell_ppc = np.arange(0,200)#np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))
    select_cell_pfc = np.arange(0,200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))

    n_trial=300

    fr_parietals = []
    fr_pfcs = []
    labels = []

    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences
    stim_dist = np.random.choice([1 * np.pi / 2, -1 * np.pi / 2],size=n_trial)
    print('pref',pref)
    print('stim_dist', stim_dist)
    random_pref = np.random.choice(pref, size=n_trial, replace=True)
    random_sign = np.random.choice([-1, 1],size=n_trial)


    for i in range(n_trial):

        gaussian_center1 = random_pref[i]
        gaussian_center2 = (gaussian_center1 + stim_dist[i]) % (2 * np.pi)


        # print('==gaussian_center1',gaussian_center1)
        # print('gaussian_center2', gaussian_center2)

        sign = random_sign[i]

        fr_parietal, fr_pfc_up = get_single_activity(model_dir, hp, batch_size=512, rule_name='retro',
                                                        gaussian_center1=gaussian_center1,
                                                        gaussian_center2=gaussian_center2,
                                                        cue_sign=1)

        fr_parietal, fr_pfc_down = get_single_activity(model_dir, hp, batch_size=512, rule_name='retro',
                                                  gaussian_center1=gaussian_center1,
                                                  gaussian_center2=gaussian_center2,
                                                  cue_sign=1)


        plt.plot(fr_pfc_up)
        plt.title('upper')
        plt.ylim([0,2])
        plt.show()

        plt.plot(fr_pfc_down)
        plt.title('lower')
        plt.ylim([0, 2])
        plt.show()
        sys.exit(0)




        # print('fr_pfc',fr_pfc.shape)

        labels.append(sign)
        fr_parietals.append(fr_parietal[:, select_cell_ppc])
        fr_pfcs.append(fr_pfc[:, select_cell_pfc])

    fr_parietals=np.array(fr_parietals)
    fr_pfcs = np.array(fr_pfcs)
    labels = np.array(labels)

    print('fr_parietals=',fr_parietals.shape)
    print('labels=',labels.shape)





    accuracy_ppcs = []
    accuracy_pfcs = []
    for time_point in range(0,stim2_off):
        accuracy_ppc = decode_data(data=fr_parietals,target=labels,t=time_point)
        accuracy_ppcs.append(accuracy_ppc)

    for time_point in range(0,stim2_off):
        accuracy_pfc = decode_data(data=fr_pfcs, target=labels, t=time_point)
        accuracy_pfcs.append(accuracy_pfc)

    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.15, hspace=0.2, wspace=0.3)
    fig.suptitle(model_name + '/' + str(idx), fontsize=9)

    # print("ppcs", accuracy_ppcs)
    # print("pfcs",accuracy_pfcs)
    plt.plot(accuracy_ppcs,c='g',label='ppc')
    plt.plot(accuracy_pfcs,'-o', c='r',label='pfc')
    axs.axvline(stim1_off, color='darkgrey', linestyle='--',label='stim_on')
    axs.axvline(stim2_on-1, color='darkgrey', linestyle='-', label='cue_on')
    axs.axvspan(stim1_off, stim2_off, color='0.95')
    plt.xlabel('from Cue_on',fontsize=15)
    plt.ylabel('classifier accuracy',fontsize=15)
    plt.legend()
    plt.savefig(figure_path + model_name+'_'+str(idx) + 'Select.png')
    plt.show()





def get_single_trial(model_dir,hp,batch_size=1,
                                rule_name='both',
                                gaussian_center1=0,
                                gaussian_center2=0,
                                cue_sign=None):

    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))

    runnerObj = run.Runner(rule_name=rule_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test')
    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            gaussian_center1=gaussian_center1,
                                            gaussian_center2=gaussian_center2,
                                            cue_sign=cue_sign)

    #### average value over batch_sizes for hidden state
    trial_idx=0
    fr_parietal = run_result.firing_rate_binder_parietal.detach().cpu().numpy()
    fr_parietal_list = list([fr_parietal[:, i, :] for i in range(batch_size)])
    fr_parietal_mean = np.mean(np.array(fr_parietal_list), axis=0)


    fr_pfc = run_result.firing_rate_binder_pfc.detach().cpu().numpy()
    fr_pfc_list = list([fr_pfc[:, i, :] for i in range(batch_size)])
    fr_pfc_mean = np.mean(np.array(fr_pfc_list), axis=0)

    return fr_parietal_mean, fr_pfc_mean


def decode_up_low_both(fig_path,model_name, model_dir, idx, hp,sigma_x=0.01):
    hp['sigma_x'] = sigma_x

    figure_path = os.path.join(fig_path, 'decode_up_low_both'+'/')
    tools.mkdir_p(figure_path)
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    print('epoch',epoch)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]


    select_cell_ppc = np.arange(0,200)#np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))
    select_cell_pfc = np.arange(0,200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))

    n_trial=300

    fr_parietals = []
    fr_pfcs = []
    labels = []

    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences
    stim_dist = np.random.choice([1 * np.pi / 2, -1 * np.pi / 2],size=n_trial)
    print('pref',pref)
    print('stim_dist', stim_dist)
    random_pref = np.random.choice(pref, size=n_trial, replace=True)
    random_sign = np.random.choice([-1, 1],size=n_trial)


    for i in range(n_trial):

        gaussian_center1 = random_pref[i]
        gaussian_center2 = (gaussian_center1 + stim_dist[i]) % (2 * np.pi)


        # print('==gaussian_center1',gaussian_center1)
        # print('gaussian_center2', gaussian_center2)

        sign = random_sign[i]

        fr_parietal, fr_pfc = get_single_trial(model_dir, hp, batch_size=1, rule_name='retro',
                                                        gaussian_center1=gaussian_center1,
                                                        gaussian_center2=gaussian_center2,
                                                        cue_sign=sign)

        # print('fr_pfc',fr_pfc.shape)

        labels.append(sign)
        fr_parietals.append(fr_parietal[:, select_cell_ppc])
        fr_pfcs.append(fr_pfc[:, select_cell_pfc])

    fr_parietals=np.array(fr_parietals)
    fr_pfcs = np.array(fr_pfcs)
    labels = np.array(labels)

    print('fr_parietals=',fr_parietals.shape)
    print('labels=',labels.shape)


    start = stim1_off
    accuracy_ppcs = []
    accuracy_pfcs = []
    for time_point in range(start,stim2_off):
        accuracy_ppc = decode_data(data=fr_parietals,target=labels,t=time_point)
        accuracy_ppcs.append(accuracy_ppc)

    for time_point in range(start,stim2_off):
        accuracy_pfc = decode_data(data=fr_pfcs, target=labels, t=time_point)
        accuracy_pfcs.append(accuracy_pfc)

    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.15, hspace=0.2, wspace=0.3)
    fig.suptitle(model_name + '/' + str(idx), fontsize=9)

    # print("ppcs", accuracy_ppcs)
    # print("pfcs",accuracy_pfcs)
    plt.plot(accuracy_ppcs,c='g',label='ppc')
    plt.plot(accuracy_pfcs,'-o', c='r',label='pfc')
    axs.axvline(stim1_off-start, color='darkgrey', linestyle='--',label='stim_on')
    axs.axvline(stim2_on-1-start, color='darkgrey', linestyle='-', label='cue_on')
    axs.axvspan(stim1_off-start, stim2_off-start, color='0.95')
    plt.xlabel('from Cue_on',fontsize=15)
    plt.ylabel('classifier accuracy',fontsize=15)
    plt.legend()
    plt.savefig(figure_path + model_name+'_'+str(idx) + 'Select.png')
    plt.show()


def get_trials(model_dir,hp,batch_size=1,
                                rule_name='both',
                                gaussian_center1=0,
                                gaussian_center2=0,
                                cue_sign=None):

    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))

    runnerObj = run.Runner(rule_name=rule_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test')
    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            gaussian_center1=gaussian_center1,
                                            gaussian_center2=gaussian_center2,
                                            cue_sign=cue_sign)

    #### average value over batch_sizes for hidden state
    trial_idx=0
    fr_parietal = run_result.firing_rate_binder_parietal.detach().cpu().numpy()
    fr_parietal_list = list([fr_parietal[:, i, :] for i in range(batch_size)])
    fr_parietal_mean = np.mean(np.array(fr_parietal_list), axis=0)


    fr_pfc = run_result.firing_rate_binder_pfc.detach().cpu().numpy()
    fr_pfc_list = list([fr_pfc[:, i, :] for i in range(batch_size)])
    fr_pfc_mean = np.mean(np.array(fr_pfc_list), axis=0)


    threshold = 1.0
    exc = int(1 * hp['n_rnn'])
    encode_ppc   = np.where(np.max(fr_parietal_mean[0:40, :exc], axis=0) > threshold)[0]
    encode_pfc = np.where(np.max(fr_pfc_mean[0:40, :exc], axis=0) > threshold)[0]







    return fr_parietal, fr_pfc,encode_ppc,encode_pfc






def get_allangle_activity_mean_space(model_dir,hp,batch_size=1,
                                rule_name='both',
                                cue_sign=None):


    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))


    # print('cue_sign',cue_sign)
    # sys.exit(0)




    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences
    fr_pfc_angles = []
    fr_parietal_angles = []
    fr_all_angles = []


    for angle in np.array(pref):

        gaussian_center1 = angle
        stim_dist = np.random.choice([1 * np.pi / 2, -1 * np.pi / 2])
        gaussian_center2 = (gaussian_center1 + stim_dist) % (2 * np.pi)#(gaussian_center1 + 1 * np.pi)% (2 * np.pi)

        runnerObj = run.Runner(rule_name=rule_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test')
        trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                                gaussian_center1=gaussian_center1,
                                                gaussian_center2=gaussian_center2,
                                                cue_sign=cue_sign)
        #### average value over batch_sizes for hidden state
        fr_pfc = run_result.firing_rate_binder_pfc.detach().cpu().numpy()
        fr_pfc_list = list([fr_pfc[:, i, :] for i in range(batch_size)])
        fr_pfc_mean = np.mean(np.array(fr_pfc_list), axis=0)
        fr_pfc_angles.append(fr_pfc_mean)

        fr_parietal = run_result.firing_rate_binder_parietal.detach().cpu().numpy()
        fr_parietal_list = list([fr_parietal[:, i, :] for i in range(batch_size)])
        fr_parietal_mean = np.mean(np.array(fr_parietal_list), axis=0)
        fr_parietal_angles.append(fr_parietal_mean)

        fr_all = run_result.firing_rate_binder.detach().cpu().numpy()
        fr_all_list = list([fr_all[:, i, :] for i in range(batch_size)])
        fr_all_mean = np.mean(np.array(fr_all_list), axis=0)
        fr_all_angles.append(fr_all_mean)


    fr_pfc_angles_array = np.array(fr_pfc_angles)
    fr_pfc_angles_mean=np.mean(np.array(fr_pfc_angles_array), axis=0)

    fr_parietal_angles_array = np.array(fr_parietal_angles)
    fr_parietal_angles_mean = np.mean(np.array(fr_parietal_angles_array), axis=0)


    fr_all_angles_array = np.array(fr_all_angles)
    fr_all_angles_mean = np.mean(np.array(fr_all_angles_array), axis=0)

    return fr_pfc_angles_mean,fr_parietal_angles_mean,fr_all_angles_mean



def find_encode_neuron_pfc(figure_path,model_dir, hp):
    batch_size = 512
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]
    # print('epoch_0', epoch_0)

    cue_on = stim2_on
    cue_off = stim2_off

    exc = int(1*hp['n_rnn'])
    start = epoch['stim2_on'][0]
    end = epoch['stim2_off'][0]

    # start = epoch['stim2_on'][0] + 10
    # end = epoch['stim2_off'][0] + 20
    #
    firing_rate_up_ret,_,_ = get_allangle_activity_mean_space(model_dir, hp, batch_size=batch_size, rule_name='retro', cue_sign=1)
    firing_rate_down_ret,_,_ = get_allangle_activity_mean_space(model_dir, hp, batch_size=batch_size, rule_name='retro', cue_sign=-1)

    threshold = 0.02 * np.max(np.mean(firing_rate_up_ret[start:end, :], axis=0))
    print('threshold:',threshold)

    mean_up = np.where(np.mean(firing_rate_up_ret[start:end, :exc], axis=0) > threshold )[0]
    mean_down = np.where(np.mean(firing_rate_down_ret[start:end, :exc], axis=0) > threshold )[0]

    print('mean_up', mean_up.shape)
    print('mean_down', mean_down.shape)
    num = np.min([mean_up.shape[0],mean_down.shape[0]])



    return mean_up[:num],mean_down[:num]




def decode_timepoint(parietal_0,parietal_1,start,end):
    accuracies = []
    for t in range(start,end):
        X_0 = parietal_0[t]
        X_1 = parietal_1[t]
        X = np.concatenate([X_0, X_1], axis=0)
        y = np.concatenate([np.ones(X_0.shape[0]), np.zeros(X_1.shape[0])])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        #clf = LogisticRegression(max_iter=1000)
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    return accuracies


def decode_up_low_both_new(fig_path,model_name, model_dir, idx, hp,sigma_x=0.01):
    hp['sigma_x'] = sigma_x

    figure_path = os.path.join(fig_path, 'decode_up_low_both'+'/')
    tools.mkdir_p(figure_path)
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    print('epoch',epoch)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]


    select_cell_ppc = np.arange(0,200)#np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))
    select_cell_pfc = np.arange(0,200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))

    n_trial=300

    fr_parietals = []
    fr_pfcs = []
    labels = []

    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences
    stim_dist = np.random.choice([1 * np.pi / 2, -1 * np.pi / 2],size=n_trial)


    batch_size=512
    rng = hp['rng']
    stim_dist = rng.choice([1 * np.pi / 2, -1 * np.pi / 2], (batch_size,))
    gaussian_center1 = rng.choice(pref, (batch_size,))
    gaussian_center2 = (gaussian_center1 + stim_dist) % (2 * np.pi)



    # print('==gaussian_center1',gaussian_center1)
    # print('gaussian_center2', gaussian_center2)



    parietal_0, pfc_0,_,_ = get_trials(model_dir, hp, batch_size=512, rule_name='retro',
                                         gaussian_center1=gaussian_center1,
                                         gaussian_center2=gaussian_center2,
                                         cue_sign = 1)

    parietal_1, pfc_1,_,_ = get_trials(model_dir, hp, batch_size=512, rule_name='retro',
                                         gaussian_center1=gaussian_center1,
                                         gaussian_center2=gaussian_center2,
                                         cue_sign = -1)


    start = stim1_off
    end = stim2_off+4
    accuracy_ppc = decode_timepoint(parietal_0,parietal_1,start,end)
    accuracy_pfc = decode_timepoint(pfc_0, pfc_1,start,end)
    print('accuracy_ppc',accuracy_ppc[stim1_off:stim2_off])
    print('accuracy_pfc', accuracy_pfc[stim1_off:stim2_off])


    # fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    # fig.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.15, hspace=0.2, wspace=0.3)
    # fig.suptitle(model_name + '/' + str(idx), fontsize=9)
    #
    # # print("ppcs", accuracy_ppcs)
    # # print("pfcs",accuracy_pfcs)
    # plt.plot(accuracy_ppc,c='g',label='ppc')
    # plt.plot(accuracy_pfc,'-o', c='r',label='pfc')
    # axs.axvline(stim1_off-start, color='darkgrey', linestyle='--',label='stim_on')
    # axs.axvline(stim2_on-1-start, color='darkgrey', linestyle='-', label='cue_on')
    # axs.axvspan(stim1_off-start, stim2_off-start, color='0.95')
    # plt.xlabel('from Cue_on',fontsize=15)
    # plt.ylabel('classifier accuracy',fontsize=15)
    # plt.legend()
    # plt.savefig(figure_path + model_name+'_'+str(idx) + 'Select.png')
    # plt.show()

    return accuracy_ppc,accuracy_pfc





def generalize_decode(parietal_retro_0,parietal_retro_1,parietal_prosp_0,parietal_prosp_1,start,end):
    accuracies = []

    times = parietal_retro_0.shape[0]

    for t in range(start, end):
        X_train = np.concatenate([parietal_retro_0[t], parietal_retro_1[t]], axis=0)
        y_train = np.concatenate([np.ones(512), np.zeros(512)])

        X_test = np.concatenate([parietal_prosp_0[t], parietal_prosp_1[t]], axis=0)
        y_test = np.concatenate([np.ones(512), np.zeros(512)])

        #clf = LogisticRegression(max_iter=1000)
        clf = LogisticRegression(penalty='l2', C=60, solver='liblinear', max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    return accuracies


def decode_up_low_both_generalize_retro2prosp(fig_path, model_name, model_dir, idx, hp):
    suffix = tools.splid_model_name(model_name, start_str='ao')
    file_name = suffix + '_' + str(idx) + '_seed' + str(hp['seed'])


    figure_path = os.path.join(fig_path, 'decode_up_low_both_generalize' + '/')
    tools.mkdir_p(figure_path)
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]

    select_cell_ppc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))
    select_cell_pfc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))


    fr_parietals = []
    fr_pfcs = []
    labels = []

    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences

    batch_size = 512
    rng = hp['rng']
    stim_dist = rng.choice([1 * np.pi / 2, -1 * np.pi / 2], (batch_size,))
    gaussian_center1 = rng.choice(pref, (batch_size,))
    gaussian_center2 = (gaussian_center1 + stim_dist) % (2 * np.pi)

    # print('==gaussian_center1',gaussian_center1)
    # print('gaussian_center2', gaussian_center2)

    parietal_retro_0, pfc_retro_0,encode_ppc_retro_0,encode_pfc_retro_0 = get_trials(model_dir, hp, batch_size=512, rule_name='retro',
                                   gaussian_center1=gaussian_center1,
                                   gaussian_center2=gaussian_center2,
                                   cue_sign=1)

    parietal_retro_1, pfc_retro_1,encode_ppc_retro_1,encode_pfc_retro_1 = get_trials(model_dir, hp, batch_size=512, rule_name='retro',
                                   gaussian_center1=gaussian_center1,
                                   gaussian_center2=gaussian_center2,
                                   cue_sign=-1)




    parietal_prosp_0, pfc_prosp_0,encode_ppc_prosp_0,encode_pfc_prosp_0 = get_trials(model_dir, hp, batch_size=512, rule_name='prosp',
                                               gaussian_center1=gaussian_center1,
                                               gaussian_center2=gaussian_center2,
                                               cue_sign=1)

    parietal_prosp_1, pfc_prosp_1,encode_ppc_prosp_1,encode_pfc_prosp_1= get_trials(model_dir, hp, batch_size=512, rule_name='prosp',
                                               gaussian_center1=gaussian_center1,
                                               gaussian_center2=gaussian_center2,
                                               cue_sign=-1)


    num_ppc = np.min([encode_ppc_retro_0.shape[0], encode_ppc_retro_1.shape[0]])
    num_pfc = np.min([encode_pfc_retro_0.shape[0], encode_pfc_retro_1.shape[0]])

    print('num_ppc,num_pfc',num_ppc,num_pfc)
    ppc_idxs = encode_ppc_retro_0[0:num_ppc]
    pfc_idxs = encode_pfc_retro_0[0:num_pfc]

    # for trial in range(10):
    #
    #     for i in range(200):
    #         plt.plot(parietal_retro_0[:,trial,i])
    #     plt.show()



    start = stim1_off+1
    end = stim2_off+4
    accuracy_ppc = generalize_decode(parietal_retro_0[:,:,ppc_idxs], parietal_retro_1[:,:,ppc_idxs],parietal_prosp_0[:,:,ppc_idxs],parietal_prosp_1[:,:,ppc_idxs],start,end)
    accuracy_pfc = generalize_decode(pfc_retro_0[:,:,pfc_idxs], pfc_retro_1[:,:,pfc_idxs],pfc_prosp_0[:,:,pfc_idxs],pfc_prosp_1[:,:,pfc_idxs],start,end)


    #
    # fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    # fig.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.15, hspace=0.2, wspace=0.3)
    # fig.suptitle(model_name + '/' + str(idx), fontsize=9)
    #
    # # print("ppcs", accuracy_ppcs)
    # # print("pfcs",accuracy_pfcs)
    # plt.plot(accuracy_ppc, c='g', label='ppc')
    # plt.plot(accuracy_pfc, '-o', c='r', label='pfc')
    # axs.axvline(stim1_off - start, color='darkgrey', linestyle='--', label='stim_on')
    # axs.axvline(stim2_on - 1 - start, color='darkgrey', linestyle='-', label='cue_on')
    # axs.axvspan(stim1_off - start, stim2_off - start, color='0.95')
    # plt.xlabel('from Cue_on', fontsize=15)
    # plt.ylabel('classifier accuracy', fontsize=15)
    # plt.legend()
    # plt.ylim([0.3,0.9])
    # plt.savefig(figure_path + file_name+'.png')
    # plt.show()

    return accuracy_ppc,accuracy_pfc



def decode_up_low_both_generalize_prosp2retro(fig_path, model_name, model_dir, idx, hp):
    suffix = tools.splid_model_name(model_name, start_str='ao')
    file_name = suffix + '_' + str(idx) + '_seed' + str(hp['seed'])


    figure_path = os.path.join(fig_path, 'decode_up_low_both_generalize' + '/')
    tools.mkdir_p(figure_path)
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]

    select_cell_ppc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))
    select_cell_pfc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))


    fr_parietals = []
    fr_pfcs = []
    labels = []

    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences

    batch_size = 512
    rng = hp['rng']
    stim_dist = rng.choice([1 * np.pi / 2, -1 * np.pi / 2], (batch_size,))
    gaussian_center1 = rng.choice(pref, (batch_size,))
    gaussian_center2 = (gaussian_center1 + stim_dist) % (2 * np.pi)

    # print('==gaussian_center1',gaussian_center1)
    # print('gaussian_center2', gaussian_center2)

    parietal_retro_0, pfc_retro_0,encode_ppc_retro_0,encode_pfc_retro_0 = get_trials(model_dir, hp, batch_size=512, rule_name='retro',
                                   gaussian_center1=gaussian_center1,
                                   gaussian_center2=gaussian_center2,
                                   cue_sign=1)

    parietal_retro_1, pfc_retro_1,encode_ppc_retro_1,encode_pfc_retro_1 = get_trials(model_dir, hp, batch_size=512, rule_name='retro',
                                   gaussian_center1=gaussian_center1,
                                   gaussian_center2=gaussian_center2,
                                   cue_sign=-1)




    parietal_prosp_0, pfc_prosp_0,encode_ppc_prosp_0,encode_pfc_prosp_0 = get_trials(model_dir, hp, batch_size=512, rule_name='prosp',
                                               gaussian_center1=gaussian_center1,
                                               gaussian_center2=gaussian_center2,
                                               cue_sign=1)

    parietal_prosp_1, pfc_prosp_1,encode_ppc_prosp_1,encode_pfc_prosp_1= get_trials(model_dir, hp, batch_size=512, rule_name='prosp',
                                               gaussian_center1=gaussian_center1,
                                               gaussian_center2=gaussian_center2,
                                               cue_sign=-1)


    num_ppc = np.min([encode_ppc_retro_0.shape[0], encode_ppc_retro_1.shape[0]])
    num_pfc = np.min([encode_pfc_retro_0.shape[0], encode_pfc_retro_1.shape[0]])

    print('num_ppc,num_pfc',num_ppc,num_pfc)
    ppc_idxs = encode_ppc_retro_0[0:num_ppc]
    pfc_idxs = encode_pfc_retro_0[0:num_pfc]

    # for trial in range(10):
    #
    #     for i in range(200):
    #         plt.plot(parietal_retro_0[:,trial,i])
    #     plt.show()



    start = stim1_off+1
    end = stim2_off+4


    accuracy_ppc = generalize_decode(parietal_prosp_0[:, :, ppc_idxs], parietal_prosp_1[:, :, ppc_idxs],
                                     parietal_retro_0[:, :, ppc_idxs], parietal_retro_1[:, :, ppc_idxs], start, end)
    accuracy_pfc = generalize_decode(pfc_prosp_0[:, :, pfc_idxs], pfc_prosp_1[:, :, pfc_idxs],
                                     pfc_retro_0[:, :, pfc_idxs], pfc_retro_1[:, :, pfc_idxs], start, end)

    #
    # fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    # fig.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.15, hspace=0.2, wspace=0.3)
    # fig.suptitle(model_name + '/' + str(idx), fontsize=9)
    #
    # # print("ppcs", accuracy_ppcs)
    # # print("pfcs",accuracy_pfcs)
    # plt.plot(accuracy_ppc, c='g', label='ppc')
    # plt.plot(accuracy_pfc, '-o', c='r', label='pfc')
    # axs.axvline(stim1_off - start, color='darkgrey', linestyle='--', label='stim_on')
    # axs.axvline(stim2_on - 1 - start, color='darkgrey', linestyle='-', label='cue_on')
    # axs.axvspan(stim1_off - start, stim2_off - start, color='0.95')
    # plt.xlabel('from Cue_on', fontsize=15)
    # plt.ylabel('classifier accuracy', fontsize=15)
    # plt.legend()
    # plt.ylim([0.3,0.9])
    # plt.savefig(figure_path + file_name+'.png')
    # plt.show()

    return accuracy_ppc,accuracy_pfc









def get_trial_one_color(model_dir,hp,batch_size=1,
                                rule_name='both',
                                gaussian_center1=0,
                                gaussian_center2=0,
                                cue_sign=None):

    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))

    runnerObj = run.Runner(rule_name=rule_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test')
    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            gaussian_center1=gaussian_center1,
                                            gaussian_center2=gaussian_center2,
                                            cue_sign=cue_sign)

    #### average value over batch_sizes for hidden state
    trial_idx=0
    fr_parietal = run_result.firing_rate_binder_parietal.detach().cpu().numpy()
    fr_parietal_list = list([fr_parietal[:, i, :] for i in range(batch_size)])
    fr_parietal_mean = np.mean(np.array(fr_parietal_list), axis=0)


    fr_pfc = run_result.firing_rate_binder_pfc.detach().cpu().numpy()
    fr_pfc_list = list([fr_pfc[:, i, :] for i in range(batch_size)])
    fr_pfc_mean = np.mean(np.array(fr_pfc_list), axis=0)
    # print('fr_pfc_mean',fr_pfc_mean.shape)


    threshold = 1.2
    exc = int(1 * hp['n_rnn'])
    encode_ppc   = np.where(np.max(fr_parietal_mean[:, :exc], axis=0) > threshold)[0]
    encode_pfc = np.where(np.max(fr_pfc_mean[:, :exc], axis=0) > threshold)[0]







    return fr_parietal, fr_pfc


def find_encode_z_score_ppc(fig_path, model_name, model_dir, idx, hp, sigma_x=0.01):
    hp['sigma_x'] = sigma_x

    figure_path = os.path.join(fig_path, 'decode_up_low_both' + '/')
    tools.mkdir_p(figure_path)
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    print('epoch', epoch)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]

    select_cell_ppc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))
    select_cell_pfc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))

    n_trial = 300

    fr_parietals = []
    fr_pfcs = []
    labels = []

    prefs = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences
    stim_dist = 1 * np.pi / 2

    batch_size = 512
    rng = hp['rng']
    gaussian_center1 = rng.choice(prefs, (batch_size,))

    # print('==gaussian_center1',gaussian_center1)
    # print('gaussian_center2', gaussian_center2)

    ppc_retro_0_list = []
    ppc_retro_1_list = []
    for angle in prefs:

        ppc_retro_0, _ = get_trial_one_color(model_dir, hp, batch_size=512, rule_name='retro',
                                       gaussian_center1=angle,
                                       gaussian_center2=angle+stim_dist,
                                       cue_sign=1)

        ppc_retro_1, _,  = get_trial_one_color(model_dir, hp, batch_size=512, rule_name='retro',
                                       gaussian_center1=angle,
                                       gaussian_center2=angle + stim_dist,
                                       cue_sign=-1)

        ppc_retro_0_list.append(ppc_retro_0)
        ppc_retro_1_list.append(ppc_retro_1)



    # -------------------------------
    # Step 1: Define MI_colour function
    # -------------------------------
    def compute_mi_colour(firing_rates, color_labels, n_bins=8, n_shuffles=100):
        """
        Compute modulation index (MI_colour) and z-score for a single neuron.
        """
        r_c = np.array([firing_rates[color_labels == c].mean() for c in range(n_bins)])
        z = r_c / (r_c.sum() + 1e-10)
        mi = (z * np.log(z * n_bins + 1e-10)).sum() / np.log(n_bins)

        # Null distribution
        null_mis = []
        for _ in range(n_shuffles):
            labels_shuf = np.random.permutation(color_labels)
            r_c_shuf = np.array([firing_rates[labels_shuf == c].mean() for c in range(n_bins)])
            z_shuf = r_c_shuf / (r_c_shuf.sum() + 1e-10)
            mi_shuf = (z_shuf * np.log(z_shuf * n_bins + 1e-10)).sum() / np.log(n_bins)
            null_mis.append(mi_shuf)

        null_mis = np.array(null_mis)
        z_score = (mi - null_mis.mean()) / (null_mis.std() + 1e-10)
        return mi, z_score

    # -------------------------------
    # Step 2: Load precomputed data arrays
    # These would be passed from user's code context
    # -------------------------------

    # Example placeholder shapes for ppc_retro_0_list and ppc_retro_1_list:
    # (8, 48, 512, 200)
    T = 48
    B = 512
    N = 200
    n_angles = 8

    # Simulated data for demonstration purposes only
    np.random.seed(0)
    ppc_retro_0_list = np.random.rand(n_angles, T, B, N)
    ppc_retro_1_list = np.random.rand(n_angles, T, B, N)

    # -------------------------------
    # Step 3: Identify selective neurons
    # -------------------------------
    t = 20  # Time index to evaluate

    t_start, t_end = stim2_on+1, stim2_on+6  # use inclusive indexing

    # Collapse over color bins
    # Average across time window
    fr_upper = ppc_retro_0_list[:, t_start:t_end + 1, :, :].mean(axis=1).reshape(-1, 200)  # shape: (8*512, 200)
    fr_lower = ppc_retro_1_list[:, t_start:t_end + 1, :, :].mean(axis=1).reshape(-1, 200)

    color_labels = np.repeat(np.arange(8), B)  # shape: (8*512,)

    upper_selective = []
    lower_selective = []
    both_selective = []

    # For plotting
    z_scores_upper = []
    z_scores_lower = []

    for n in range(N):
        fr_u = fr_upper[:, n]
        fr_l = fr_lower[:, n]

        mi_u, z_u = compute_mi_colour(fr_u, color_labels)
        mi_l, z_l = compute_mi_colour(fr_l, color_labels)

        z_scores_upper.append(z_u)
        z_scores_lower.append(z_l)

        if z_u > 1.96 and z_l > 1.96:
            both_selective.append(n)
        elif z_u > 1.96:
            upper_selective.append(n)
        elif z_l > 1.96:
            lower_selective.append(n)

    print('upper_selective',upper_selective)
    print('lower_selective', lower_selective)
    print('both_selective',both_selective)


    return upper_selective+both_selective, lower_selective+both_selective


def find_encode_z_score_pfc(fig_path, model_name, model_dir, idx, hp, sigma_x=0.01):
    hp['sigma_x'] = sigma_x

    figure_path = os.path.join(fig_path, 'decode_up_low_both' + '/')
    tools.mkdir_p(figure_path)
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    print('epoch', epoch)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]

    select_cell_ppc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))
    select_cell_pfc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))

    n_trial = 300

    fr_parietals = []
    fr_pfcs = []
    labels = []

    prefs = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences
    stim_dist = 1 * np.pi / 2

    batch_size = 512
    rng = hp['rng']
    gaussian_center1 = rng.choice(prefs, (batch_size,))

    # print('==gaussian_center1',gaussian_center1)
    # print('gaussian_center2', gaussian_center2)

    ppc_retro_0_list = []
    ppc_retro_1_list = []
    for angle in prefs:

        _, ppc_retro_0 = get_trial_one_color(model_dir, hp, batch_size=512, rule_name='retro',
                                       gaussian_center1=angle,
                                       gaussian_center2=angle+stim_dist,
                                       cue_sign=1)

        _, ppc_retro_1,  = get_trial_one_color(model_dir, hp, batch_size=512, rule_name='retro',
                                       gaussian_center1=angle,
                                       gaussian_center2=angle + stim_dist,
                                       cue_sign=-1)

        ppc_retro_0_list.append(ppc_retro_0)
        ppc_retro_1_list.append(ppc_retro_1)



    # -------------------------------
    # Step 1: Define MI_colour function
    # -------------------------------
    def compute_mi_colour(firing_rates, color_labels, n_bins=8, n_shuffles=100):
        """
        Compute modulation index (MI_colour) and z-score for a single neuron.
        """
        r_c = np.array([firing_rates[color_labels == c].mean() for c in range(n_bins)])
        z = r_c / (r_c.sum() + 1e-10)
        mi = (z * np.log(z * n_bins + 1e-10)).sum() / np.log(n_bins)

        # Null distribution
        null_mis = []
        for _ in range(n_shuffles):
            labels_shuf = np.random.permutation(color_labels)
            r_c_shuf = np.array([firing_rates[labels_shuf == c].mean() for c in range(n_bins)])
            z_shuf = r_c_shuf / (r_c_shuf.sum() + 1e-10)
            mi_shuf = (z_shuf * np.log(z_shuf * n_bins + 1e-10)).sum() / np.log(n_bins)
            null_mis.append(mi_shuf)

        null_mis = np.array(null_mis)
        z_score = (mi - null_mis.mean()) / (null_mis.std() + 1e-10)
        return mi, z_score

    # -------------------------------
    # Step 2: Load precomputed data arrays
    # These would be passed from user's code context
    # -------------------------------

    # Example placeholder shapes for ppc_retro_0_list and ppc_retro_1_list:
    # (8, 48, 512, 200)
    T = 48
    B = 512
    N = 200
    n_angles = 8

    # Simulated data for demonstration purposes only
    np.random.seed(0)
    ppc_retro_0_list = np.random.rand(n_angles, T, B, N)
    ppc_retro_1_list = np.random.rand(n_angles, T, B, N)

    # -------------------------------
    # Step 3: Identify selective neurons
    # -------------------------------
    t = 20  # Time index to evaluate

    # Collapse over color bins
    t_start, t_end = stim2_on, stim2_off  # use inclusive indexing

    # Collapse over color bins
    # Average across time window
    fr_upper = ppc_retro_0_list[:, t_start:t_end + 1, :, :].mean(axis=1).reshape(-1, 200)  # shape: (8*512, 200)
    fr_lower = ppc_retro_1_list[:, t_start:t_end + 1, :, :].mean(axis=1).reshape(-1, 200)


    color_labels = np.repeat(np.arange(8), B)  # shape: (8*512,)

    upper_selective = []
    lower_selective = []
    both_selective = []

    # For plotting
    z_scores_upper = []
    z_scores_lower = []

    for n in range(N):
        fr_u = fr_upper[:, n]
        fr_l = fr_lower[:, n]

        mi_u, z_u = compute_mi_colour(fr_u, color_labels)
        mi_l, z_l = compute_mi_colour(fr_l, color_labels)

        z_scores_upper.append(z_u)
        z_scores_lower.append(z_l)

        if z_u > 1.96 and z_l > 1.96:
            both_selective.append(n)
        elif z_u > 1.96:
            upper_selective.append(n)
        elif z_l > 1.96:
            lower_selective.append(n)

    print('upper_selective',upper_selective)
    print('lower_selective', lower_selective)


    return upper_selective, lower_selective



def decode_up_low_both_generalize1(fig_path, model_name, model_dir, idx, hp, sigma_x=0.01):
    hp['sigma_x'] = sigma_x

    figure_path = os.path.join(fig_path, 'decode_up_low_both' + '/')
    tools.mkdir_p(figure_path)
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    print('epoch', epoch)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]

    select_cell_ppc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))
    select_cell_pfc = np.arange(0, 200)  # np.random.randint(0,200,100)#np.arange(20)##int(1*int(hp['n_rnn'] / 2))

    n_trial = 300

    fr_parietals = []
    fr_pfcs = []
    labels = []

    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences


    batch_size = 512
    rng = hp['rng']
    stim_dist = rng.choice([1 * np.pi / 2, -1 * np.pi / 2], (batch_size,))
    gaussian_center1 = rng.choice(pref, (batch_size,))
    gaussian_center2 = (gaussian_center1 + stim_dist) % (2 * np.pi)


    # print('==gaussian_center1',gaussian_center1)
    # print('gaussian_center2', gaussian_center2)

    parietal_retro_0, pfc_retro_0,encode_ppc_retro_0,encode_pfc_retro_0 = get_trials(model_dir, hp, batch_size=512, rule_name='retro',
                                   gaussian_center1=gaussian_center1,
                                   gaussian_center2=gaussian_center2,
                                   cue_sign=1)

    parietal_retro_1, pfc_retro_1,encode_ppc_retro_1,encode_pfc_retro_1 = get_trials(model_dir, hp, batch_size=512, rule_name='retro',
                                   gaussian_center1=gaussian_center1,
                                   gaussian_center2=gaussian_center2,
                                   cue_sign=-1)




    parietal_prosp_0, pfc_prosp_0,encode_ppc_prosp_0,encode_pfc_prosp_0 = get_trials(model_dir, hp, batch_size=512, rule_name='prosp',
                                               gaussian_center1=gaussian_center1,
                                               gaussian_center2=gaussian_center2,
                                               cue_sign=1)

    parietal_prosp_1, pfc_prosp_1,encode_ppc_prosp_1,encode_pfc_prosp_1= get_trials(model_dir, hp, batch_size=512, rule_name='prosp',
                                               gaussian_center1=gaussian_center1,
                                               gaussian_center2=gaussian_center2,
                                               cue_sign=-1)

    # print('encode_ppc_retro_0,encode_pfc_retro_0 ',encode_ppc_retro_0,encode_pfc_retro_0 )
    # print('encode_ppc_retro_1, encode_pfc_retro_1 ', encode_ppc_retro_1, encode_pfc_retro_1)
    #
    # print('encode_ppc_prosp_0, encode_pfc_prosp_0', encode_ppc_prosp_0, encode_pfc_prosp_0)
    # print('encode_ppc_prosp_1, encode_pfc_prosp_1', encode_ppc_prosp_1, encode_pfc_prosp_1)

    upper_selective_ppc, lower_selective_ppc = find_encode_z_score_ppc(fig_path, model_name, model_dir, idx, hp, sigma_x=0.01)
    #upper_selective_pfc, lower_selective_pfc = find_encode_z_score_pfc(fig_path, model_name, model_dir, idx, hp, sigma_x=0.01)
    upper_selective_pfc, lower_selective_pfc = upper_selective_ppc, lower_selective_ppc
    num_ppc = np.min([len(upper_selective_ppc), len(lower_selective_ppc)])
    num_pfc = np.min([len(upper_selective_pfc), len(lower_selective_pfc)])


    ppc_idxs = encode_ppc_retro_0[0:num_ppc]
    pfc_idxs = encode_pfc_retro_0[0:num_pfc]

    # for trial in range(10):
    #
    #     for i in range(200):
    #         plt.plot(parietal_retro_0[:,trial,i])
    #     plt.show()



    start = stim1_off
    end = stim2_off
    accuracy_ppc = generalize_decode(parietal_retro_0[:,:,ppc_idxs], parietal_retro_1[:,:,ppc_idxs],parietal_prosp_0[:,:,ppc_idxs],parietal_prosp_1[:,:,ppc_idxs],start,end)
    accuracy_pfc = generalize_decode(pfc_retro_0[:,:,pfc_idxs], pfc_retro_1[:,:,pfc_idxs],pfc_prosp_0[:,:,pfc_idxs],pfc_prosp_1[:,:,pfc_idxs],start,end)



    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.15, hspace=0.2, wspace=0.3)
    fig.suptitle(model_name + '/' + str(idx), fontsize=9)

    # print("ppcs", accuracy_ppcs)
    # print("pfcs",accuracy_pfcs)
    plt.plot(accuracy_ppc, c='g', label='ppc')
    plt.plot(accuracy_pfc, '-o', c='r', label='pfc')
    axs.axvline(stim1_off - start, color='darkgrey', linestyle='--', label='stim_on')
    axs.axvline(stim2_on - 1 - start, color='darkgrey', linestyle='-', label='cue_on')
    axs.axvspan(stim1_off - start, stim2_off - start, color='0.95')
    plt.xlabel('from Cue_on', fontsize=15)
    plt.ylabel('classifier accuracy', fontsize=15)
    plt.legend()
    plt.ylim([0.4,1])
    plt.savefig(figure_path + model_name + '_' + str(idx) + 'Select.png')
    plt.show()





































