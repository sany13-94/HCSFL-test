import socket
import time
import sys
import traceback

import numpy as np

from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauServer
from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg, get_indices_each_node_case

# ---- Config
from config import *

# ----------------- helpers (better diagnostics) -----------------
def safe_send(sock, payload, tag):
    try:
        send_msg(sock, payload)
    except Exception as e:
        print(f"[server] send to {tag} failed: {e}")
        raise

def safe_recv(sock, expect_tag, tag):
    try:
        return recv_msg(sock, expect_tag)
    except Exception as e:
        print(f"[server] recv from {tag} failed: {e}")
        raise

# ----------------- model init -----------------
model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)

use_fixed_averaging_slots = (time_gen is not None)

# Preload dataset when using SGD (batch < total_data)
if batch_size < total_data:
    train_image, train_label, test_image, test_label, train_label_orig = get_data(
        dataset, total_data, dataset_file_path
    )
    # deterministic partitioning prepared once
    indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

# ----------------- listen & accept -----------------
listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_ADDR, SERVER_PORT))

client_sock_all = []

# Make backlog generous so bursty connects don't fail
listening_sock.listen(64)
print(f"[server] Listening on {SERVER_ADDR}:{SERVER_PORT}")

# Accept up to n_nodes clients
while len(client_sock_all) < n_nodes:
    print("Waiting for incoming connections...")
    client_sock, (ip, port) = listening_sock.accept()
    idx = len(client_sock_all)
    client_sock_all.append(client_sock)
    print(f"Got connection from {ip}:{port} -> client#{idx}")

# Stats writer
if single_run:
    stat = CollectStatistics(results_file_name=single_run_results_file_path, is_single_run=True)
else:
    stat = CollectStatistics(results_file_name=multi_run_results_file_path, is_single_run=False)

try:
    for sim in sim_runs:

        # For full-batch runs (batch_size >= total_data), reload each sim
        if batch_size >= total_data:
            train_image, train_label, test_image, test_label, train_label_orig = get_data(
                dataset, total_data, dataset_file_path, sim_round=sim
            )
            indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

        for case in case_range:
            for tau_setup in tau_setup_all:

                stat.init_stat_new_global_round()

                dim_w = model.get_weight_dimension(train_image, train_label)
                w_global_init = model.get_init_weight(dim_w, rand_seed=sim)
                w_global = w_global_init.copy()

                w_global_min_loss = None
                loss_min = np.inf
                prev_loss_is_min = False

                if tau_setup < 0:
                    is_adapt_local = True
                    tau_config = 1
                else:
                    is_adapt_local = False
                    tau_config = tau_setup

                if is_adapt_local or estimate_beta_delta_in_all_runs:
                    if tau_setup == -1:
                        control_alg = ControlAlgAdaptiveTauServer(
                            is_adapt_local, dim_w, client_sock_all, n_nodes,
                            control_param_phi, moving_average_holding_param
                        )
                    else:
                        raise Exception('Invalid setup of tau.')
                else:
                    control_alg = None

                # ----------- INIT message to every client -----------
                for n in range(n_nodes):
                    indices_this_node = indices_each_node_case[case][n]

                    # Optional domain assignment for PATHMNIST_DOMAINS
                    if dataset == 'PATHMNIST_DOMAINS':
                        domain_id = 3 if (n == n_nodes - 1) else (n % 3)
                    else:
                        domain_id = None

                    use_control_alg = (control_alg is not None)  # boolean flag

                    msg_init = [
                        'MSG_INIT_SERVER_TO_CLIENT',
                        model_name, dataset,
                        num_iterations_with_same_minibatch_for_tau_equals_one, step_size,
                        batch_size, total_data,
                        use_control_alg,                      # 7
                        indices_this_node, read_all_data_for_stochastic,
                        use_min_loss, sim, domain_id, n       # ... up to 13
                    ]
                    print(f"[server] Sending INIT to client#{n} | data_size={len(indices_this_node)} | use_ctrl={use_control_alg}")
                    safe_send(client_sock_all[n], msg_init, f"client#{n}")

                print('All clients connected / initialized.')

                # Barrier: wait each client to finish local data prep
                for n in range(n_nodes):
                    safe_recv(client_sock_all[n], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER', f"client#{n}")

                print('Start learning')

                time_global_aggregation_all = None
                total_time = 0.0
                total_time_recomputed = 0.0
                it_each_local = None
                it_each_global = None
                is_last_round = False
                is_eval_only = False
                tau_new_resume = None

                # ----------- federated rounds -----------
                while True:
                    print('---------------------------------------------------------------------------')
                    print('current tau config:', tau_config)

                    time_total_all_start = time.time()

                    # broadcast (w_global, tau, ...)
                    for n in range(n_nodes):
                        safe_send(
                            client_sock_all[n],
                            ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau_config, is_last_round, prev_loss_is_min],
                            f"client#{n}"
                        )

                    w_global_prev = w_global.copy()

                    print('Waiting for local iteration at client')

                    w_global = np.zeros(dim_w, dtype=np.float32)
                    loss_last_global = 0.0
                    loss_w_prev_min_loss = 0.0
                    received_loss_local_w_prev_min_loss = False
                    data_size_total = 0
                    time_all_local_all = 0.0
                    data_size_local_all = []

                    tau_actual = 0

                    # gather
                    for n in range(n_nodes):
                        msg = safe_recv(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', f"client#{n}")
                        # ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, tau_actual, data_size_local,
                        #  loss_last_global, loss_w_prev_min_loss]
                        w_local = msg[1]
                        time_all_local = msg[2]
                        tau_actual = max(tau_actual, msg[3])
                        data_size_local = msg[4]
                        loss_local_last_global = msg[5]
                        loss_local_w_prev_min_loss = msg[6]

                        w_global += w_local * data_size_local
                        data_size_local_all.append(data_size_local)
                        data_size_total += data_size_local
                        time_all_local_all = max(time_all_local_all, float(time_all_local))

                        if use_min_loss:
                            loss_last_global += loss_local_last_global * data_size_local
                            if loss_local_w_prev_min_loss is not None:
                                loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
                                received_loss_local_w_prev_min_loss = True

                    w_global /= max(1, data_size_total)

                    if True in np.isnan(w_global):
                        print('*** w_global is NaN, using previous value')
                        w_global = w_global_prev
                        use_w_global_prev_due_to_nan = True
                    else:
                        use_w_global_prev_due_to_nan = False

                    if use_min_loss:
                        loss_last_global /= max(1, data_size_total)
                        if received_loss_local_w_prev_min_loss:
                            loss_w_prev_min_loss /= max(1, data_size_total)
                            loss_min = loss_w_prev_min_loss

                        if loss_last_global < loss_min:
                            loss_min = loss_last_global
                            w_global_min_loss = w_global_prev
                            prev_loss_is_min = True
                        else:
                            prev_loss_is_min = False

                        print("Loss of previous global value:", loss_last_global)
                        print("Minimum loss:", loss_min)

                    time_total_all_end = time.time()
                    time_total_all = time_total_all_end - time_total_all_start
                    time_global_aggregation_all = max(0.0, time_total_all - time_all_local_all)

                    print('Time for one local iteration:', (time_all_local_all / max(1, tau_actual)))
                    print('Time for global averaging:', time_global_aggregation_all)

                    if use_fixed_averaging_slots:
                        if isinstance(time_gen, (list, tuple)):
                            t_g = time_gen[case]
                        else:
                            t_g = time_gen
                        it_each_local = max(1e-8, np.sum(t_g.get_local(tau_actual)) / max(1, tau_actual))
                        it_each_global = float(t_g.get_global(1)[0])
                    else:
                        it_each_local = max(1e-8, time_all_local_all / max(1, tau_actual))
                        it_each_global = time_global_aggregation_all

                    total_time_recomputed += it_each_local * tau_actual + it_each_global
                    total_time += time_total_all

                    stat.collect_stat_end_local_round(
                        case, tau_actual, it_each_local, it_each_global, control_alg, model,
                        train_image, train_label, test_image, test_label, w_global,
                        total_time_recomputed
                    )

                    # plan next tau
                    if not use_w_global_prev_due_to_nan:
                        if control_alg is not None:
                            tau_new = control_alg.compute_new_tau(
                                data_size_local_all, data_size_total,
                                it_each_local, it_each_global, max_time,
                                step_size, tau_config, use_min_loss
                            )
                        else:
                            if tau_new_resume is not None:
                                tau_new = tau_new_resume
                                tau_new_resume = None
                            else:
                                tau_new = tau_config
                    else:
                        if tau_new_resume is None:
                            tau_new_resume = tau_config
                        tau_new = 1

                    # resource budget guard
                    is_last_round_tmp = False
                    if use_min_loss:
                        tmp_remain = total_time_recomputed + it_each_local * (tau_new + 1) + 2 * it_each_global
                    else:
                        tmp_remain = total_time_recomputed + it_each_local * tau_new + it_each_global

                    if tmp_remain < max_time:
                        tau_config = tau_new
                    else:
                        if use_min_loss:
                            tau_config = int((max_time - total_time_recomputed - 2 * it_each_global - it_each_local) / it_each_local)
                        else:
                            tau_config = int((max_time - total_time_recomputed - it_each_global) / it_each_local)

                        if tau_config < 1:
                            tau_config = 1
                        elif tau_config > tau_new:
                            tau_config = tau_new

                        is_last_round_tmp = True

                    if is_last_round:
                        break

                    if is_eval_only:
                        tau_config = 1
                        is_last_round = True

                    if is_last_round_tmp:
                        if use_min_loss:
                            is_eval_only = True
                        else:
                            is_last_round = True

                # evaluation weights
                w_eval = w_global_min_loss if use_min_loss else w_global

                stat.collect_stat_end_global_round(
                    sim, case, tau_setup, total_time, model, train_image, train_label,
                    test_image, test_label, w_eval, total_time_recomputed
                )

except Exception as e:
    print("[server] Fatal error:", e)
    traceback.print_exc()
finally:
    for i, cs in enumerate(client_sock_all):
        try:
            cs.close()
            print(f"[server] closed client#{i}")
        except Exception:
            pass
    try:
        listening_sock.close()
    except Exception:
        pass
    print("[server] shutdown complete")