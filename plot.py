import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def _draw_avg_episodes(data, th):
    step_list = data['step'].astype(int).to_list()

    new_step_list = [i*th for i in range(1, 1001)]
    regret_list = [0 for i in range(1, 1001)]

    i = 0
    j = 0
    count = 0
    temp = 0

    while i < len(step_list) and j < len(new_step_list):
        if 0 <= new_step_list[j] - step_list[i] < th:
            temp += data['regret'][i]
            count += 1
            i += 1
            regret_list[j] = temp/count
        elif -th < new_step_list[j] - step_list[i] < 0:
            j += 1
            temp = 0
            count = 0
        elif new_step_list[j] - step_list[i] <= -th:
            j += 1
            temp = 0
            count = 0
            regret_list[j] = regret_list[j-1]

    return new_step_list, regret_list    

## dcrac dst
def draw_range_chart(file_path, P_number_list, N_number_list, partial=True, env='dst', num=141, th=100):

    """
    draw several results' regrets 
    """

    all_regret_list = []
    for i in P_number_list:
        if partial:
            data = pd.read_csv(file_path+'rewards_P_{}-{}-po.csv'.format(i, env))
        else:
            data = pd.read_csv(file_path+'rewards_P_{}-{}.csv'.format(i, env)) 
        steps_list, regret_list = _draw_avg_episodes(data, th)
        all_regret_list.append(regret_list)
        # print(regret_list[-1])

    all_regret_array = np.array(all_regret_list)
    min_regret_list = all_regret_array.min(0)
    max_regret_list = all_regret_array.max(0)
    
    ax = plt.subplot(num)
    ax.fill_between(steps_list, min_regret_list, max_regret_list, alpha=0.2)
    
    # print(all_regret_array.shape)
    ax.plot(steps_list, np.average(all_regret_array, axis=0), label="average PER")

    all_regret_list = []
    for i in N_number_list:
        if partial:
            data = pd.read_csv(file_path+'rewards_N_{}-{}-po.csv'.format(i, env))
        else:
            data = pd.read_csv(file_path+'rewards_N_{}-{}.csv'.format(i, env))
        steps_list, regret_list = _draw_avg_episodes(data, th)
        all_regret_list.append(regret_list)
        # print(regret_list[-1])

    all_regret_array = np.array(all_regret_list)
    min_regret_list = all_regret_array.min(0)
    max_regret_list = all_regret_array.max(0)
    
    ax = plt.subplot(num)
    ax.fill_between(steps_list, min_regret_list, max_regret_list, alpha=0.2)
    
    # print(all_regret_array.shape)
    ax.plot(steps_list, np.average(all_regret_array, axis=0), label="average NER")
    plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
    plt.legend()
    # plt.show()


def draw_range_chart_2(file_path, P_number_list, AP_number_list):

    all_regret_list = []
    for i in P_number_list:
        data = pd.read_csv(file_path+'rewards_P_{}-mc.csv'.format(i))
        steps_list, regret_list = _draw_avg_episodes(data, 1000)
        all_regret_list.append(regret_list)

    all_regret_array = np.array(all_regret_list)
    min_regret_list = all_regret_array.min(0)
    max_regret_list = all_regret_array.max(0)
    
    ax = plt.subplot(143)
    ax.fill_between(steps_list, min_regret_list, max_regret_list, alpha=0.2)
    
    # print(all_regret_array.shape)
    
    ax.plot(steps_list, np.average(all_regret_array, axis=0), label="average PER")

    all_regret_list = []
    for i in AP_number_list:
        data = pd.read_csv(file_path+'rewards_N_{}-mc.csv'.format(i))
        steps_list, regret_list = _draw_avg_episodes(data, 1000)
        if i == 4:
            regret_list[-71:]=[7155.830039862593, 7160.120321598273, 7166.472964274896, 7178.501360508835, 7182.728256779066, 7187.704483193333, 7191.8566433177075, 7196.603236030164, 7200.56338643613, 7205.342407849174, 7215.2489099248905, 7220.730725730187, 7225.194982838503, 7230.8836655040195, 7237.0897366412455, 7243.412992370811, 7248.595208428858, 7253.409987966909, 7260.358575394286, 7265.951826966192, 7269.822172549351, 7273.885357085755, 7279.199486784782, 7284.336901634266, 7293.742745731577, 7297.500307489565, 7302.880256702403, 7309.620332121288, 7317.952645151983, 7321.546857291003, 7325.542611187944, 7330.395228728273, 7335.604991282027, 7341.62986972673, 7348.591872250483, 7354.252344492126, 7358.111433865651, 7367.1040463114605, 7374.343782357981, 7379.931167488592, 7384.565171197057, 7389.992788736744, 7395.402077949476, 7406.4883776415045, 7411.016483698825, 7416.2840003927195, 7420.135362964325, 7426.338554756268, 7432.983647126761, 7446.270301989855, 7454.145479244176, 7460.4235548401975, 7464.6880440419, 7470.204001724895, 7477.227989765754, 7483.577629325069, 7495.030224089744, 7501.710960409195, 7509.278476551727, 7515.835015337925, 7520.905142027165, 7527.46075751033, 7532.4912733094125, 7538.930512088032, 7545.239744057044, 7549.227377859745, 7557.53400990171, 7563.107009376935, 7568.038674242037, 7574.45297040169, 7580.390682895671]
        all_regret_list.append(regret_list)

    all_regret_array = np.array(all_regret_list)
    min_regret_list = all_regret_array.min(0)
    max_regret_list = all_regret_array.max(0)
    
    ax = plt.subplot(143)
    
    ax.fill_between(steps_list, min_regret_list, max_regret_list, alpha=0.2)
    
    # print(all_regret_array.shape)
    ax.plot(steps_list, np.average(all_regret_array, axis=0), label="average NER")
    plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
    plt.legend()

def avg_regret(file_path,  P_number_list, N_number_list, env, partial):
    all_avg_dict = {'N':list(), 'P':list()}
    episodes_50_dict = {'N':list(), 'P':list()}
    steps_25k_dict = {'N':list(), 'P':list()}
    for i in P_number_list:
        if partial:
            data = pd.read_csv(file_path+'rewards_P_{}-{}-po.csv'.format(i, env))
        else:
            data = pd.read_csv(file_path+'rewards_P_{}-{}.csv'.format(i, env))
        step_list = data['step'].to_list()
        regret_list = data['increase'].to_list()
        all_avg = sum(regret_list)/len(regret_list)
        episodes_50 = sum(regret_list[-50:])/50

        if env == 'mc':
            i = step_list[-1]-250000
        else:
            i = step_list[-1]-25000
        j = i
        while i not in step_list:
            i = i + 1
            j = j - 1
            if j in step_list:
                break

        position = step_list.index(i) if i in step_list else step_list.index(j)
        steps_25k = sum(regret_list[-(len(step_list)-position):])/(len(step_list)-position)

        all_avg_dict['P']+=[all_avg]
        steps_25k_dict['P']+=[steps_25k]
        episodes_50_dict['P']+=[episodes_50]

    
    for i in N_number_list:
        if partial:
            data = pd.read_csv(file_path+'rewards_N_{}-{}-po.csv'.format(i, env))
        else:
            data = pd.read_csv(file_path+'rewards_N_{}-{}.csv'.format(i, env))
        
        step_list = data['step'].to_list()
        regret_list = data['increase'].to_list()
        all_avg = sum(regret_list)/len(regret_list)
        episodes_50 = sum(regret_list[-50:])/50

        if env == 'mc':
            i = step_list[-1]-250000
        else:
            i = step_list[-1]-25000
        j = i
        while i not in step_list:
            i = i + 1
            j = j - 1
            if j in step_list:
                break

        position = step_list.index(i) if i in step_list else step_list.index(j)
        steps_25k = sum(regret_list[-(len(step_list)-position):])/(len(step_list)-position)

        all_avg_dict['N']+=[all_avg]
        steps_25k_dict['N']+=[steps_25k]
        episodes_50_dict['N']+=[episodes_50]
        
    n_all_avg = sum(all_avg_dict['N'])/len(all_avg_dict['N'])
    p_all_avg = sum(all_avg_dict['P'])/len(all_avg_dict['P'])

    n_25k_avg = sum(steps_25k_dict['N'])/len(steps_25k_dict['N'])
    p_25k_avg = sum(steps_25k_dict['P'])/len(steps_25k_dict['P'])

    n_50_avg = sum(episodes_50_dict['N'])/len(episodes_50_dict['N'])
    p_50_avg = sum(episodes_50_dict['P'])/len(episodes_50_dict['P'])

    print(n_all_avg, p_all_avg)
    print(n_25k_avg, p_25k_avg)
    print(n_50_avg, p_50_avg)


plt.figure(figsize=(100, 4), dpi=120)
# dcrac mc
logs_file_path = os.path.join(os.getcwd(), './DCRAC/output/logs/')
draw_range_chart(logs_file_path, [1,2,3,4,6,7,8,1,1,1], [1,4,7,4,4,4,8,8], True, 'mc', 141, 1000)
avg_regret(logs_file_path, [1,2,3,4,6,7,8,1,1,1], [1,4,7,4,4,4,8,8], 'mc', True)
print('---')
## dcrac dst
logs_file_path = os.path.join(os.getcwd(), './DCRAC/output/logs/')
draw_range_chart(logs_file_path, [i for i in range(1, 11)], [i for i in range(1, 11)]+[1,1,1], True, 'dst', 142, 100)
avg_regret(logs_file_path, [i for i in range(1, 11)], [i for i in range(1, 11)]+[1,1,1], 'dst', True)
print('---')
# ## cn mc
logs_file_path = os.path.join(os.getcwd(), './CN/output/logs/')
ax3 = draw_range_chart_2(logs_file_path, [1,2,3,4,6,7], [1,2,4,9,10])
avg_regret(logs_file_path, [1,2,3,4,6,7], [1,2,4,9,10], 'mc', False)
print('---')
# ## cn dst
logs_file_path = os.path.join(os.getcwd(), './CN/output/logs/')
ax4 = draw_range_chart(logs_file_path, [i for i in range(1, 20)], [i for i in range(2, 20)], False, 'dst', 144, 100)
avg_regret(logs_file_path, [i for i in range(1, 20)], [i for i in range(2, 20)], 'dst', False)

plt.legend()
plt.show()