
from gurobipy import *
import numpy as np
from numpy.random.mtrand import f
from pyswarms.single.global_best import GlobalBestPSO
from scipy.optimize import minimize
def calculate_trans_based_on_comp(mu, y):
    # 基于计算时隙分配，得到计算时延，基于计算时延，优化目标不变，求解传输时隙分配
    # 构建模型
    m = Model("T_tran_LP")
    m.setParam('OutputFlag', 0)
    # 定义变量
    x = m.addVars(n_t, n_k, n_t, name="x",lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    T_compS = np.zeros((n_t, n_k), dtype=np.int32) # 计算开始时刻，约束条件
    OBJ = 0
    # 对于任意一个任务t_prime，计算其在t_prime + tau时刻的完成率
    for t_prime in range(n_t):
        for k in range(n_k):
            for j in range(n_j):
                if mu[t_prime, k, j] == 0: continue
                # 计算总共需要的计算数据量
                tmp_nz = np.where(y[t_prime, k, :, j] != 0)[0]
                # task_compS是y[t_prime, k, :]中第一个非零元素的下标，即任务开始计算的时刻
                task_compS = tmp_nz[0]
                T_compS[t_prime, k] = task_compS+1
                # 总时延
                totalT = quicksum(x[t_prime, k, t] * (t-t_prime) for t in range(t_prime, T_compS[t_prime, k])) # 变量，优化目标是每个时隙的工作时长*时隙的idx.这样的好处是1*t<=0.5*t+0.5*(t+1),因此会尽量分配到前面的时隙,也就是最小化总时延
                OBJ += totalT #/ (tau[t_prime, k])
    m.setObjective(OBJ, GRB.MINIMIZE) # 最小化总时延
    for t_prime in range(n_t):
        for k in range(n_k):
            if up[t_prime,k] == 0 or sum(mu[t_prime, k, :]) == 0:
                continue # 没有传输任务，不需要计算
            for j in range(n_j):
                if mu[t_prime, k, j] == 0:
                    continue
                # C14.1
                m.addConstr((B / n_k) * quicksum(x[t_prime, k, tp] * g[k, j, tp] for tp in range(t_prime, T_compS[t_prime, k])) * dt >= up[t_prime, k], f"C14.1_{t_prime}_{k}_{j}") # 在t_prime到T_compS[t_prime, k]时刻，传输的数据量大于等于任务的数据量
    m.optimize()
    # print(m.runtime)
    if m.status == GRB.Status.OPTIMAL:
        optimal_x = m.getAttr('x', x)
        # 返回x变量的求解数值
        sol_x = np.zeros((n_t, n_k, n_t))
        for (i, j, k), value in optimal_x.items():
            sol_x[i, j, k] = value
        return sol_x, m.objVal
    return None, None

def calculate_comp_based_on_trans(mu, x):
    m = Model("T_comp_LP")
    m.setParam('OutputFlag', 0)
    # 定义变量
    y = m.addVars(n_t, n_k, n_t, n_j, name="y",lb=0.0, ub=F_j, vtype=GRB.CONTINUOUS)
    T_tranE = n_t * np.ones((n_t, n_k), dtype=np.int32) # 计算开始时刻，约束条件
    OBJ = 0
    # 对于任意一个任务t_prime，计算其在t_prime + tau时刻的完成率
    for t_prime in range(n_t):
        for k in range(n_k):
            for j in range(n_j):
                if mu[t_prime, k, j] != 0:
                    assert tau[t_prime, k]!=0
                    task_tranE = np.where(x[t_prime, k, :] != 0)[0][-1]
                    T_tranE[t_prime, k] = task_tranE
                    # 总时延
                    totalT = quicksum(y[t_prime, k, t, j] * (t-t_prime) for t in range(T_tranE[t_prime, k], min(n_t,t_prime+tau[t_prime,k]+1)))
                    OBJ += totalT# / (tau[t_prime, k]) # 让梯度控制的有差距
    m.setObjective(OBJ, GRB.MINIMIZE)
    for t_prime in range(n_t):
        for j in range(n_j):
            m.addConstr(quicksum(y[tp, k, t_prime, j] for tp in range(n_t) for k in range(n_k)) <= F_j, f"C14.2_{t_prime}_{j}")
        for k in range(n_k):
            for j in range(n_j):
                if mu[t_prime, k, j] == 0:
                    continue
                # C15
                m.addConstr(quicksum(y[t_prime, k, tp, j] for tp in range(T_tranE[t_prime, k], min(n_t,t_prime+tau[t_prime,k]+1))) * dt >= req[t_prime, k], f"C15.1_{t_prime}_{k}_{j}")

    m.optimize()
    # print(m.runtime)
    if m.status == GRB.Status.OPTIMAL:
        optimal_y = m.getAttr('x', y)
        # 返回x变量的求解数值
        sol_y = np.zeros((n_t, n_k, n_t, n_j))
        for (i, j, k, l), value in optimal_y.items():
            sol_y[i, j, k, l] = value

        return sol_y, m.objVal
    return None, None
def initialize_trans(mu):
    # 根据mu的值，初始化x,直接使用贪心算法
    x = np.zeros((n_t, n_k, n_t))
    tmp_g = g.copy()
    for t_prime in range(n_t):
        for k in range(n_k):
            if sum(mu[t_prime, k, :]) == 0:
                continue
            j = np.argmax(mu[t_prime, k, :])
            data_sz = up[t_prime, k]
            band = B / n_k
            for t in range(t_prime, n_t):
                x[t_prime, k, t] = 1
                data_sz -= band * dt * tmp_g[k, j, t]
                tmp_g[k, j, t] = 0
                if data_sz <= 0:
                    break
            if data_sz > 0:
                mu[t_prime, k, j] = 0
    return x

def cal_obj_v(x, y, mu):
    # x代表时隙t内占据了0.x个子时隙用于传输, n_t * n_k * n_t
    # y代表时隙t内占据了多少的计算资源, n_t * n_k * n_t * n_j
    # mu代表了是否将任务t_prime传输到设备k上, n_t * n_k * n_j
    obj = 0
    succ_task_cnt = 0
    for t_prime in range(n_t):
        for k in range(n_k):
            if up[t_prime, k] == 0:
                continue
            # 判断是否有任务,如果有,并且mu全是0,则添加一个M的惩罚
            if sum(mu[t_prime, k, :]) == 0:
                obj += 2
                continue
            succ_task_cnt += 1
            for j in range(n_j):
                if mu[t_prime, k, j] == 0:
                    continue
                # 总时延是y的最后一个非零项
                tranE_idx = np.where(x[t_prime, k, :] != 0)
                compE_idx = np.where(y[t_prime, k, :, j] != 0)
                
                task_compE = compE_idx[0][-1]
                task_compE = (task_compE+x[t_prime, k, tranE_idx[0][-1]]) * dt
                if isBaseStation and j + 10 > n_j:
                    task_compE += (up[t_prime, k] / (150 / n_k)) 
                    
                if isAllBS and j + 20 > n_j:
                    task_compE += (up[t_prime, k] / (150 / n_k)) 

                obj += task_compE - t_prime * dt
    # print("succ_task_cnt:", succ_task_cnt)
    return obj, succ_task_cnt
def ao_based_on_mu(mu, max_iterations=100, tolerance=1e-6):
    # 初始化变量
    x = initialize_trans(mu)  # 一个你自己定义的初始化x的函数
    y = np.zeros((n_t, n_k, n_t, n_j))  # 全0
    hist_objVal = []
    max_objVal = 0
    # 交替优化的循环
    for i in range(max_iterations):
        # 保存前一次迭代的变量，以便后续检查收敛
        x_prev = x.copy()
        y_prev = y.copy()

        # 第一步：固定x，优化y
        y, objVal1 = calculate_comp_based_on_trans(mu, x)
        if y is None:
            return 0, 1000, 0
        # 第二步：固定y，优化x
        x, objVal2 = calculate_trans_based_on_comp(mu, y)
        if x is None:
            return 0, 1000, 0
        hist_objVal.append(objVal1+objVal2)
        max_objVal = max(max_objVal, objVal1 + objVal2)
        # 检查收敛：如果x和y的变化很小，则停止  
        break
        if np.linalg.norm(x - x_prev) < tolerance and np.linalg.norm(y - y_prev) < tolerance:
            # print(f"Converged in {i+1} iterations")
            break
        # else:
    # if i == max_iterations - 1:
    #     print("Max iterations reached")
    lat, ratio = cal_obj_v(x, y, mu)
    # 返回最终的x和y
    return max_objVal, lat, ratio

# val, sec = ao_based_on_mu(assigned_matrix)
# print(sec)
# 匈牙利算法
def get_ass_by_Hungarian():
    assigned_matrix = np.zeros((n_t, n_k, n_j))
    # F_j: [n_j], res_j:[n_j, n_t], 把F_j扩充即可
    res_j = np.repeat(F_j[:, np.newaxis], n_t, axis=1)
    window_size = 5
    tmp_up = up.copy()
    tot_sec = 0
    succ_task = 0
    tmp_g = g.copy()
    sum_task_num_in_window = np.zeros(int(n_t//window_size)+1)
    for s in range(0, n_t, window_size): # 每5个时隙进行一次分配
        window_no = int(s // window_size)
        sum_task_num_in_window[window_no] = len(np.where(tmp_up[s:min(s+window_size, n_t),:]!=0))
        while True:
            tot_cnt = np.where(tmp_up[s:s+window_size, :] != 0)[0].shape[0]
            cur_cnt = 0
            if tot_cnt == 0:
                break
            # 创建代价矩阵
            cost_matrix = np.zeros((tot_cnt, n_j))
            # 记录任务索引，用于在任务分配之后更新任务分配矩阵
            task_indices = {}
            for t in range(s, min(s+window_size, n_t)): # 对于每个时间窗口内的时隙
                for k in range(n_k):
                    if tmp_up[t, k] != 0: # 如果有任务
                        for j in range(n_j):
                            # 初始设备j的资源
                            # cost_time =  (5 - (t - s) + 1) * dt
                            cost_time = 0
                            tmp_data = 0
                            cost_trans_dt = 0
                            cost_comp_dt = 0
                            for t_t in range(t, n_t):
                                this_slot = B * tmp_g[k, j, t_t] * dt / sum_task_num_in_window[window_no] 
                                tmp_data += this_slot  # dt时间内传输的数据量
                                if tmp_data >= tmp_up[t, k]:
                                    cost_trans_dt += (tmp_up[t, k] - (tmp_data-this_slot)) / this_slot # 以dt为单位的传输时间
                                    break
                                cost_trans_dt += 1
                            # if isBaseStation and j + 10 > n_j:
                            #     t_t += int(tmp_up[t, k] / (150 / n_k) / dt) # 额外的有线传输时间
                            # if isAllBS and j + 20 > n_j:
                            #     t_t += int(tmp_up[t, k] / (300 / n_k) / dt)
                            c_t = n_t
                            tmp_comp = 0
                            for c_t in range(t_t+1, min(t+tau[t, k]+1, n_t)):
                                this_slot = res_j[j, c_t] * dt
                                tmp_comp += this_slot
                                if tmp_comp >= req[t, k]:
                                    cost_comp_dt += (req[t, k] - (tmp_comp - this_slot)) / this_slot
                                    break
                                cost_comp_dt += 1
                            cost_time = (cost_comp_dt + cost_trans_dt) * dt 
                            cost_matrix[cur_cnt, j] = (cost_time) / (tau[t, k] * dt) 
                            # 记录当前任务的索引
                            task_indices[cur_cnt] = (t, k)
                        cur_cnt += 1
            # 进行任务分配
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
            off_flag = False
            for i, j in zip(row_ind, col_ind):
                t, k = task_indices[i]
                tmp_data = 0
                flag = False
                for t_t in range(t, n_t):
                    tmp_data += B * tmp_g[k, j, t_t] * dt / sum_task_num_in_window[window_no] 
                    if tmp_data >= tmp_up[t, k]:
                        flag = True
                        break
                    
                # if isBaseStation and j + 10 > n_j:
                #     t_t += int(tmp_up[t, k] / (150 / n_k) / dt) # 额外的有线传输时间
                # if isAllBS and j + 20 > n_j:
                #     t_t += int(tmp_up[t, k] / (300 / n_k) / dt)
                    
                if t_t >= t + tau[t, k] or not flag or t_t >= n_t-1:
                    continue
                tmp_comp = 0
                flag = False
                c_t = n_t # 初始化
                for c_t in range(t_t, min(t + tau[t, k]+1, n_t)):
                    tmp_comp += res_j[j, c_t] * dt
                    if tmp_comp >= req[t, k]:
                        flag = True
                        break
                if not flag or c_t > t+tau[t, k] or c_t >= n_t-1:
                    continue # 如果计算时间大于任务的ddl，则不分配
                assigned_matrix[t, k, j] = 1
                off_flag = True
                tmp_up[t,k] = 0
                tmp_g[k,:, t:t_t+1] = 0 # 同一时刻只能传输一个任务
                # 计算任务的结束时刻
                start_time = t_t
                end_time = c_t
                # 从任务开始时刻到结束时刻，将设备j的资源置为0
                res_j[j, start_time:end_time+1] = 0
                tot_sec += (end_time+1 - t) * 0.1
                succ_task += 1
            if not off_flag:
                break
    return assigned_matrix, tot_sec, succ_task

def get_ass_by_greedy():
    assigned_matrix = np.zeros((n_t, n_k, n_j))
    res_j = np.ones((n_j, n_t)) * F_j # 代表了设备j在t时刻的计算资源
    tmp_up = up.copy()
    tmp_g = g.copy()
    tot_sec = 0
    succ_task = 0
    for t_prime in range(n_t):
        for k in range(n_k):
            if tmp_up[t_prime, k] != 0:
                # 贪心选择时间最短的进行占用
                min_time = np.inf
                min_idx = None
                min_t_t = 0
                min_c_t = 0
                for j in range(n_j):
                    data_size = tmp_up[t_prime, k]
                    flag = False
                    for t_t in range(t_prime, n_t):
                        data_size -= B/n_k * tmp_g[k, j, t_t] * dt
                        if data_size <= 0:
                            flag = True
                            break
                    if isBaseStation and j + 10 > n_j:
                        t_t += int(tmp_up[t_prime, k] / (150 / n_k) / dt) # 额外的有线传输时间
                    if isAllBS and j + 20 > n_j:
                        t_t += int(tmp_up[t_prime, k] / (300 / n_k) / dt)
                    if t_t >= t_prime + tau[t_prime, k] or not flag or t_t >= n_t-1:
                        continue
                    req_cpu = req[t_prime, k]
                    flag = False
                    c_t = n_t
                    for c_t in range(t_t+1, n_t):
                        req_cpu -= res_j[j, c_t] * dt
                        if req_cpu <= 0:
                            flag = True
                            break
                    if not flag:continue
                    if c_t < min(t_prime+tau[t_prime, k], n_t) and min_time > c_t:
                        min_time = c_t
                        min_idx = j
                        min_t_t = t_t
                        min_c_t = c_t
                if min_idx is not None:
                    assigned_matrix[t_prime, k, min_idx] = 1
                    tmp_g[k, :, t_prime:min_t_t+1] = 0
                    res_j[min_idx, min_t_t+1:min_c_t+1] = 0
                    tot_sec += (min_time+1-t_prime) * 0.1
                    tmp_up[t_prime,k] = 0
                    succ_task += 1
    return assigned_matrix, tot_sec, succ_task

from scipy.optimize import linear_sum_assignment


def WHO(n_kt = 5, n_jt = 10, n_tt = 20, F_jt = 2.3*np.ones(10), upt = None, reqt = None, taut = None, gt = None, dtt = 0.1 , Bt = 10, isBS=False, isALL=False):
    '''n_k: 任务车辆的数量
    n_j: 服务车辆的数量
    n_t: step的数量
    F_j: 服务车辆的计算资源(假设每个服务车辆的计算资源相同)
    up: 任务的数据量(n_t, n_k), up[t, k]代表了任务车辆k在t时刻的数据量,如果为0,则代表该step该车辆没有生成任务
    req: 任务的计算资源(n_t, n_k), req[t, k]代表了任务车辆k在t时刻的计算资源需求
    tau: 任务的ddl(n_t, n_k), tau[t, k]代表了任务车辆k在t时刻的ddl
    g: 任务的数据传输速率(n_k, n_j, n_t), g[k, j, t]代表了任务车辆k在t时刻将数据传输给服务车辆j的速率,不包含带宽
    dt: 时间间隔
    B: 带宽
    isBS: 是否是基站,如果是则需要考虑有线传输的时间
    '''
    # 设置变量为global
    global n_k, n_j, n_t, F_j, up, req, tau, g, dt, B, isBaseStation, isAllBS
    assert len(F_jt) == n_jt
    F_jt = np.array(F_jt)
    # 初始化变量
    n_k, n_j, n_t, F_j, dt, B, isBaseStation = n_kt, n_jt, n_tt, F_jt, dtt, Bt, isBS
    isAllBS = isALL
    if isALL:
        isBaseStation = False
    up, req, tau, g = upt, reqt, taut, gt
    ass, tot_sec, succ_task = get_ass_by_Hungarian()
    # ass, tot_sec, succ_task = get_ass_by_greedy()
    return ass
    # val, sec, succ_task = ao_based_on_mu(ass)
    # return sec, succ_task
    
if __name__ == '__main__':
    n_k = 20
    n_j = 10
    n_t = 20
    F_j = 2.3
    tot_run_time = 0
    dt = 0.1 # 时间间隔
    B = 10 # 带宽
    M = n_t * 2 # 惩罚系数
    up = np.zeros((n_t, n_k))
    req = np.zeros((n_t, n_k))
    tau = np.zeros((n_t, n_k), dtype=np.int32)
    task_cnt = 0
    # 设置随机数种子
    np.random.seed(2021)
    assigned_matrix = np.zeros((n_t, n_k, n_j))
    # 随机生成任务
    task_num_step = int(5 * n_k / n_t)
    for t in range(n_t):
        # 随机4个设备,0~n_k-1
        k_list = np.random.choice(n_k, task_num_step, replace=False)
        for k in k_list:
            up[t, k] = np.random.rand()*0.1 + 0.05 # 任务传输量, 0.02-0.2
            req[t, k] = np.random.rand()*0.2 + 0.1 # 任务计算量, 0.1-0.3
            tau[t, k] = int((np.random.rand()* 0.2 + 0.2) / dt) # 任务计算时间, 0.2-0.4，转换为时间间隔，整数
            task_cnt += 1
            
    g = np.random.rand(n_k, n_j, n_t) * 1 + 1 # 代表了设备k到达设备j在t时刻的传输速率, byte/s, 1~2 Mb/s
    sec, succ_task = WHO(n_kt = n_k, n_jt = n_j, n_tt = n_t, F_jt=F_j, upt=up, reqt=req, taut=tau, gt=g, dtt=dt, Bt=B, isBS=False, isALL=False)
    
    print(sec, succ_task)