import socket
import time
import struct
import sys
import traceback

from control_algorithm.adaptive_tau import ControlAlgAdaptiveTauClient
from data_reader.data_reader import get_data, get_data_train_samples
from models.get_model import get_model
from util.sampling import MinibatchSampling
from util.utils import send_msg, recv_msg

# ---- Config
from config import SERVER_ADDR, SERVER_PORT, dataset_file_path

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_ADDR, SERVER_PORT))
print(f"[client] Connected to {SERVER_ADDR}:{SERVER_PORT}")
print('---------------------------------------------------------------------------')

batch_size_prev = None
total_data_prev = None
sim_prev = None

try:
    while True:
        # INIT from server
        msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
        # ['MSG_INIT_SERVER_TO_CLIENT',
        #   1:model_name, 2:dataset, 3:num_iter_same_mb_for_tau1, 4:step_size, 5:batch_size,
        #   6:total_data, 7:use_control_alg(bool), 8:indices_this_node,
        #   9:read_all_data_for_stochastic, 10:use_min_loss, 11:sim, 12:domain_id, 13:client_id]

        model_name = msg[1]
        dataset = msg[2]
        num_iterations_with_same_minibatch_for_tau_equals_one = msg[3]
        step_size = msg[4]
        batch_size = msg[5]
        total_data = msg[6]
        use_control_alg = msg[7]  # bool
        indices_this_node = msg[8]
        read_all_data_for_stochastic = msg[9]
        use_min_loss = msg[10]
        sim = msg[11]
        domain_id = msg[12] if len(msg) > 12 else None
        client_id = msg[13] if len(msg) > 13 else None

        # Model(s)
        model = get_model(model_name)
        model2 = get_model(model_name)  # for loss on w_prev_min_loss (keeps model state clean)

        if hasattr(model, 'create_graph'):
            model.create_graph(learning_rate=step_size)
        if hasattr(model2, 'create_graph'):
            model2.create_graph(learning_rate=step_size)

        # Control algo (client side)
        control_alg = ControlAlgAdaptiveTauClient() if use_control_alg else None

        # Data prefetch if needed
        if read_all_data_for_stochastic or batch_size >= total_data:
            if (batch_size_prev != batch_size) or (total_data_prev != total_data) or \
               ((batch_size >= total_data) and (sim_prev != sim)):
                print('[client] Reading all data used in training...')
                train_image, train_label, _, _, _ = get_data(dataset, total_data, dataset_file_path, sim_round=sim)

        batch_size_prev = batch_size
        total_data_prev = total_data
        sim_prev = sim

        if batch_size >= total_data:
            sampler = None
            train_indices = indices_this_node
        else:
            sampler = MinibatchSampling(indices_this_node, batch_size, sim)
            train_indices = None  # set later
        last_batch_read_count = None

        data_size_local = len(indices_this_node)
        w_prev_min_loss = None
        w_last_global = None
        total_iterations = 0

        # Ack init complete
        send_msg(sock, ['MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER'])

        # ===================== training rounds =====================
        while True:
            print('---------------------------------------------------------------------------')

            msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
            # ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau, is_last_round, prev_loss_is_min]
            w = msg[1]
            tau_config = msg[2]
            is_last_round = msg[3]
            prev_loss_is_min = msg[4]

            if prev_loss_is_min or ((w_prev_min_loss is None) and (w_last_global is not None)):
                w_prev_min_loss = w_last_global

            if control_alg is not None:
                control_alg.init_new_round(w)

            time_local_start = time.time()

            grad = None
            loss_last_global = None
            loss_w_prev_min_loss = None
            tau_actual = 0

            for i in range(tau_config):

                # Mini-batch loading (when using SGD)
                if batch_size < total_data:
                    # keep first minibatch aligned across rounds for control alg signal
                    if (not isinstance(control_alg, ControlAlgAdaptiveTauClient)) or (i != 0) or (train_indices is None) or \
                       (tau_config <= 1 and
                        (last_batch_read_count is None or
                         last_batch_read_count >= num_iterations_with_same_minibatch_for_tau_equals_one)):

                        sample_indices = sampler.get_next_batch()

                        if read_all_data_for_stochastic:
                            train_indices = sample_indices
                        else:
                            train_image, train_label = get_data_train_samples(
                                dataset, sample_indices, dataset_file_path
                            )
                            train_indices = range(0, min(batch_size, len(train_label)))

                        last_batch_read_count = 0

                    last_batch_read_count += 1

                # compute grad given current weights
                grad = model.gradient(train_image, train_label, w, train_indices)

                if i == 0:
                    try:
                        loss_last_global = model.loss_from_prev_gradient_computation()
                        print('*** Loss computed from previous gradient computation')
                    except Exception:
                        loss_last_global = model.loss(train_image, train_label, w, train_indices)
                        print('*** Loss computed directly')

                    w_last_global = w

                    if use_min_loss and (batch_size < total_data) and (w_prev_min_loss is not None):
                        loss_w_prev_min_loss = model2.loss(train_image, train_label, w_prev_min_loss, train_indices)

                # Local SGD step
                w = w - step_size * grad

                tau_actual += 1
                total_iterations += 1

                if control_alg is not None:
                    is_last_local = control_alg.update_after_each_local(i, w, grad, total_iterations)
                    if is_last_local:
                        break

            time_all_local = time.time() - time_local_start
            print('time_all_local =', time_all_local)

            # Optional straggler simulation
            if (client_id in (0, 1)):
                import random
                import config as cfg
                base = float(getattr(cfg, "delay_base_sec", 10.0))
                jitter = float(getattr(cfg, "delay_jitter_sec", 3.0))
                delay = base + random.uniform(0.0, jitter)
                print(f"[client {client_id}] Simulating straggler delay: {delay:.2f}s")
                time.sleep(delay)

            if control_alg is not None:
                control_alg.update_after_all_local(model, train_image, train_label, train_indices,
                                                   w, w_last_global, loss_last_global)

            # send back result
            send_msg(sock, [
                'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER',
                w, time_all_local, tau_actual, data_size_local,
                loss_last_global, loss_w_prev_min_loss
            ])

            if control_alg is not None:
                control_alg.send_to_server(sock)

            if is_last_round:
                break

except (struct.error, socket.error):
    print('Server has stopped')
    pass
except Exception as e:
    print("[client] Fatal error:", e)
    traceback.print_exc()
finally:
    try:
        sock.close()
    except Exception:
        pass
    print("[client] shutdown complete")