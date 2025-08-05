import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
import os
import time
import csv
import threading
import traceback
import re
import queue

import scipy.io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.signal import butter, filtfilt
import librosa
import librosa.display

from predict_backend import predict_all, predict_from_array, denoise_signal

# -----------------------------
# Global State
# -----------------------------
X_loaded = None
Y_loaded = None
time_exp = None
sensor_positions = []
render_log = [] 
exp_signal_batches = []
plot_data_queue = queue.Queue()
last_processed_batch = None
snapshot_last_plotted_num = -1
batch_counter = 1

# -----------------------------
# Utilities
# -----------------------------
def highpass_filter(data, cutoff=100, fs=1000, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return filtfilt(b, a, data)

# -----------------------------
# Simulation Functions
# -----------------------------
def load_mat_file():
    global X_loaded, Y_loaded
    file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    if not file_path:
        return
    try:
        mat = scipy.io.loadmat(file_path)
        X_loaded = mat['X_n']
        Y_loaded = mat['Y_n'].reshape(-1)
        messagebox.showinfo("Loaded", f"Loaded {X_loaded.shape[0]} pipelines.")
    except Exception as e:
        messagebox.showerror("File Load Error", f"Could not read the .mat file.\n{e}")

def render_waveform(auto=False, pipeline_idx_override=None):
    global X_loaded
    if X_loaded is None:
        if not auto: messagebox.showerror("Error", "No data loaded.")
        return
    try:
        pipeline_idx = pipeline_idx_override if pipeline_idx_override is not None else int(pipeline_var.get()) - 1
        sensor_idx = 0 
        signal = X_loaded[pipeline_idx, sensor_idx, :]
        plt.figure(figsize=(8, 4))
        plt.plot(signal)
        plt.title(f"Pipeline {pipeline_idx+1} - Sensor {sensor_idx+1}")
        plt.xlabel("Time Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        if not auto: messagebox.showerror("Plot Error", traceback.format_exc())

def show_confusion_matrix():
    global X_loaded, Y_loaded
    if X_loaded is None or Y_loaded is None:
        messagebox.showerror("Error", "No data loaded.")
        return
    result = predict_all(X_loaded, Y_loaded)
    cm, f1, acc = result['confusion_matrix'], result['f1_score'], result['accuracy']
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix\nF1 Score: ${f1:.3f}$, Accuracy: ${acc:.3f}$")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(6)); ax.set_yticks(np.arange(6))
    ax.set_xticklabels([str(i) for i in range(6)]); ax.set_yticklabels([str(i) for i in range(6)])
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout(); plt.show()

def render_3d_pipeline(auto=False, pipeline_idx_override=None):
    global X_loaded, Y_loaded
    if X_loaded is None or Y_loaded is None:
        if not auto: messagebox.showerror("Error", "No data loaded.")
        return
    try:
        pipeline_idx = pipeline_idx_override if pipeline_idx_override is not None else int(pipeline_var.get()) - 1
        signal_matrix = X_loaded[pipeline_idx]
        result = predict_from_array(signal_matrix[np.newaxis, :, :])
        label = result["prediction_index"]
        label_names = ["No Damage", "Local Corrosion", "General Corrosion", "Clamp", "Weld", "Pitting"]
        predicted_label_text = label_names[label]
        energies = [np.sum(signal_matrix[i] ** 2) if i not in [0, 45] else -1 for i in range(46)]
        valid_energy_values = [e for e in energies if e != -1]
        mean_energy = np.mean(valid_energy_values)
        threshold = mean_energy * 1.4
        valid_energies = [(i, e) for i, e in enumerate(energies) if e > threshold]
        likely_damage_sensor = max(valid_energies, key=lambda x: x[1])[0] if valid_energies else None

        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(111, projection='3d')
        num_sensors, pipe_radius, pipe_length = 46, 5, 1000
        theta, x = np.linspace(0, 2 * np.pi, 50), np.linspace(0, pipe_length, num_sensors)
        z_grid, theta_grid = np.meshgrid(x, theta)
        ax.plot_surface(z_grid, pipe_radius * np.cos(theta_grid), pipe_radius * np.sin(theta_grid), color='lightgray', alpha=0.5)
        for i in range(num_sensors):
            color = 'red' if likely_damage_sensor is not None and i == likely_damage_sensor else 'blue'
            ax.scatter(x[i], pipe_radius, 0, color=color, s=80)
            ax.text(x[i], pipe_radius * 1.5, 0, f"{i+1}", size=8)

        damage_text = f"Likely Damage at Sensor {likely_damage_sensor + 1}" if likely_damage_sensor is not None else "No High-Energy Damage Detected"
        ax.set_title(f"Pipeline {pipeline_idx+1} - Predicted: {predicted_label_text}\n{damage_text}")
        ax.view_init(elev=15, azim=120); ax.axis('off')
        red_dot = plt.Line2D([0], [0], marker='o', color='w', label='Likely Damage', mfc='red', ms=10)
        blue_dot = plt.Line2D([0], [0], marker='o', color='w', label='Normal Sensor', mfc='blue', ms=10)
        ax.legend(handles=[red_dot, blue_dot], loc='upper left')
        plt.tight_layout(); plt.show()

        log_label = f"Simulation Pipeline {pipeline_idx + 1}"
        if not any(d['label'] == log_label for d in render_log):
            log_entry = {'type': 'sim', 'id': pipeline_idx, 'label': log_label}
            render_log.append(log_entry)
            log_listbox.insert(tk.END, log_label)
    except Exception as e:
        if not auto: messagebox.showerror("3D Error", traceback.format_exc())

def predict_simulation_pipeline():
    global X_loaded
    if X_loaded is None:
        messagebox.showerror("Error", "No simulation data loaded.")
        return
    try:
        pipeline_idx = int(pipeline_var.get()) - 1
        if not (0 <= pipeline_idx < X_loaded.shape[0]):
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid Input", f"Please enter a valid pipeline number between 1 and {X_loaded.shape[0]}.")
        return
    try:
        signal_matrix = X_loaded[pipeline_idx]
        input_tensor = signal_matrix[np.newaxis, :, :]
        result = predict_from_array(input_tensor)
        messagebox.showinfo(
            "Prediction Result",
            f"Prediction for Pipeline {pipeline_idx + 1}:\n\n"
            f"Label: {result['prediction_name']} (Class {result['prediction_index']})"
        )
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")

# -----------------------------
# Watcher Threads & Exp. Analysis
# -----------------------------
def watch_input_folder():
    folder, last_seen = "./input_folder", None
    while True:
        try:
            files = [f for f in os.listdir(folder) if f.endswith(".mat")]
            if files:
                latest = max(files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))
                if latest != last_seen:
                    mat = scipy.io.loadmat(os.path.join(folder, latest))
                    globals().update(X_loaded=mat['X_n'], Y_loaded=mat['Y_n'].reshape(-1), last_seen=latest)
                    print(f"Auto-loaded {latest}")
        except Exception as e: print("Watcher error:", e)
        time.sleep(5)

def watch_experimental_folder(root_folder="exp_input_folder", poll_interval=2.0):
    global exp_signal_batches
    seen_batches = set()
    while True:
        try:
            sensor_folders = sorted([f.path for f in os.scandir(root_folder) if f.is_dir()], key=lambda p: float(re.search(r'(\d+(\.\d+)?)', os.path.basename(p)).group(1) if re.search(r'(\d+(\.\d+)?)', os.path.basename(p)) else float('inf')))
            if not sensor_folders: time.sleep(poll_interval); continue
            sensor_csvs = {}
            for sensor_folder in sensor_folders:
                subdirs = [f.path for f in os.scandir(sensor_folder) if f.is_dir()]
                if not subdirs: break
                inner_folder = subdirs[0]
                sensor_csvs[sensor_folder] = {f: os.path.join(inner_folder, f) for f in sorted([f for f in os.listdir(inner_folder) if f.endswith(".csv")])}
            all_batch_ids = [set(re.search(r"_\d+", f).group() for f in files) for files in sensor_csvs.values()]
            common_batches = set.intersection(*all_batch_ids) if all_batch_ids else set()
            for batch_id in sorted(list(common_batches - seen_batches)):
                batch_data = []
                for folder, file_map in sensor_csvs.items():
                    file = next((f for f in file_map if batch_id in f), None)
                    if file is None: break
                    with open(file_map[file], 'r', newline='') as f:
                        reader = csv.reader(f)
                        numeric_rows = []
                        for row in reader:
                            if len(row) < 2: continue
                            try:
                                num_row = [float(row[0]), float(row[1])]
                                numeric_rows.append(num_row)
                            except ValueError:
                                continue
                        if numeric_rows:
                            batch_data.append(np.array(numeric_rows, dtype=float).T)
                if len(batch_data) == len(sensor_csvs):
                    min_len = min(d.shape[1] for d in batch_data)
                    full_batch = np.concatenate([d[:, :min_len] for d in batch_data], axis=0)
                    exp_signal_batches.append(full_batch)
                    seen_batches.add(batch_id)
                    print(f"[Watcher] Loaded batch {batch_id} with shape {full_batch.shape}")
                    time.sleep(0.5)
        except Exception:
             print("Error in experimental watcher:")
             traceback.print_exc()
        time.sleep(poll_interval)

def open_2d_pipeline_viewer():
    if not exp_signal_batches:
        messagebox.showerror("No Data", "No experimental batches have been loaded yet.")
        return
    
    batch_index_str = simpledialog.askstring("Select Batch", f"Enter batch number (0 to {len(exp_signal_batches)-1}):")
    if batch_index_str is None: return

    try:
        batch_index = int(batch_index_str)
        if not (0 <= batch_index < len(exp_signal_batches)): raise ValueError
    except (ValueError, TypeError):
        messagebox.showerror("Invalid Input", "Please enter a valid batch number.")
        return

    batch = exp_signal_batches[batch_index]
    n_channels, _ = batch.shape
    sensor_positions = [4.5, 12, 35, 60, 83, 107, 131, 144, 162, 180]
    if len(sensor_positions) != n_channels:
        messagebox.showerror("Mismatch", f"Expected {len(sensor_positions)} channels but got {n_channels}")
        return
    denoised_signals = {i: denoise_signal(batch[i]) for i in range(n_channels)}
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.scatter(sensor_positions, [0]*n_channels, c='blue', s=60, label="Sensor")
    ax.set(xlim=(0, 192), ylim=(-1, 1), title=f"Batch #{batch_index}: Click near a sensor to view its signal", xlabel="Position (inches)", yticks=[])
    ax.legend()
    
    def onclick(event):
        if event.xdata is None or event.inaxes != ax: return
        distances = [abs(event.xdata - p) for p in sensor_positions]
        closest_sensor_idx = np.argmin(distances)
        pos = sensor_positions[closest_sensor_idx]
        signal = denoised_signals[closest_sensor_idx]
        plt.figure(figsize=(10, 4))
        plt.plot(signal, label=f"Denoised signal at {pos}\"")
        plt.title(f"Sensor at {pos}\" (Batch {batch_index})")
        plt.xlabel("Time Index"); plt.ylabel("Amplitude"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout(); plt.show()

def predict_exp_data():
    if not exp_signal_batches:
        messagebox.showerror("No Data", "No experimental batches have been loaded yet.")
        return
    batch_index_str = simpledialog.askstring("Select Batch", f"Enter batch number (0 to {len(exp_signal_batches)-1}):")
    if batch_index_str is None: return
    try:
        batch_index = int(batch_index_str)
        if not (0 <= batch_index < len(exp_signal_batches)): raise ValueError
    except (ValueError, TypeError):
        messagebox.showerror("Invalid Input", "Please enter a valid batch number.")
        return
    batch = exp_signal_batches[batch_index]
    current_channels, current_length = batch.shape
    target_shape = (46, 1200)
    reshaped_batch = np.zeros(target_shape, dtype=np.float32)
    ch_to_copy = min(current_channels, target_shape[0])
    len_to_copy = min(current_length, target_shape[1])
    reshaped_batch[:ch_to_copy, :len_to_copy] = batch[:ch_to_copy, :len_to_copy]
    input_tensor = reshaped_batch[np.newaxis, :, :]
    try:
        result = predict_from_array(input_tensor)
        messagebox.showinfo("Prediction", f"For Batch #{batch_index}, Predicted Label is:\n{result['prediction_name']} (Class {result['prediction_index']})")
    except Exception as e: messagebox.showerror("Prediction Error", traceback.format_exc())

# -----------------------------
# Background Worker & Plotting Functions
# -----------------------------
def process_data_for_plot():
    """
    This function now also pre-calculates the prediction for each batch.
    """
    global batch_counter, last_processed_batch
    next_batch_to_process = 0 
    while True:
        if next_batch_to_process < len(exp_signal_batches):
            batch = exp_signal_batches[next_batch_to_process]
            
            n_channels, total_samples = batch.shape
            adjusted_time = np.linspace(0, total_samples / 1000, total_samples)
            denoised_batch_data = np.array([denoise_signal(ch) for ch in batch])
            
            # --- PREDICTION LOGIC MOVED TO BACKGROUND ---
            # Format data for prediction
            target_shape = (46, 1200)
            reshaped_batch = np.zeros(target_shape, dtype=np.float32)
            ch_to_copy = min(n_channels, target_shape[0])
            len_to_copy = min(total_samples, target_shape[1])
            reshaped_batch[:ch_to_copy, :len_to_copy] = batch[:ch_to_copy, :len_to_copy]
            input_tensor = reshaped_batch[np.newaxis, :, :]
            
            # Run prediction
            try:
                prediction_result = predict_from_array(input_tensor)
                prediction_text = prediction_result['prediction_name']
            except Exception as e:
                prediction_text = "Error"
                print(f"Prediction error in background thread: {e}")
            # ---------------------------------------------
            
            processed_data = {
                "batch": batch, 
                "denoised_batch": denoised_batch_data,
                "time": adjusted_time,
                "n_channels": n_channels, 
                "batch_num": batch_counter,
                "batch_index": next_batch_to_process,
                "prediction": prediction_text # Add prediction to the data packet
            }
            
            plot_data_queue.put(processed_data)
            last_processed_batch = processed_data 
            
            next_batch_to_process += 1
            batch_counter += 1
        else:
            time.sleep(0.1)

def update_waterfall_plot(root, ax, canvas, im_obj):
    latest_data_to_plot = None
    try:
        while True:
            latest_data_to_plot = plot_data_queue.get_nowait()
    except queue.Empty:
        pass

    if latest_data_to_plot:
        batch, adjusted_time = latest_data_to_plot["batch"], latest_data_to_plot["time"]
        n_channels, current_batch_num = latest_data_to_plot["n_channels"], latest_data_to_plot["batch_num"]
        
        normalized_batch = batch.copy()
        for i in range(n_channels):
            mean, std = np.mean(normalized_batch[i]), np.std(normalized_batch[i])
            if std > 1e-8:
                normalized_batch[i] = (normalized_batch[i] - mean) / std
        
        max_time = float(max_time_var.get()) if max_time_var.get() else 2.5
        mask = adjusted_time <= max_time
        
        if np.any(mask):
            plot_batch, plot_time = normalized_batch[:, mask], adjusted_time[mask]
            
            if switch_axes_var.get():
                plot_array, extent = plot_batch.T, [1, n_channels, plot_time.min(), plot_time.max()]
                ax.set_xlabel('Sensor Index'); ax.set_ylabel('Time (s)')
            else:
                plot_array, extent = plot_batch, [plot_time.min(), plot_time.max(), 1, n_channels]
                ax.set_xlabel('Time (s)'); ax.set_ylabel('Sensor Index')
            
            im_obj.set_data(plot_array)
            im_obj.set_extent(extent)
            abs_max = np.max(np.abs(plot_array)) if plot_array.size > 0 else 1
            im_obj.set_clim(vmin=-abs_max, vmax=abs_max)
            ax.set_title(f'Live Waterfall - Batch #{current_batch_num} (0–{max_time:.2f}s)')
            
            interpolation_style = 'bilinear' if interpolation_enabled.get() else 'nearest'
            im_obj.set_interpolation(interpolation_style)
            canvas.draw()
            
    root.after(100, update_waterfall_plot, root, ax, canvas, im_obj)

def update_snapshot_waveform_plot(root, ax, canvas, channel_selector):
    global snapshot_last_plotted_num
    
    force_redraw = [False] 
    def on_select(event=None):
        force_redraw[0] = True
    channel_selector.bind("<<ComboboxSelected>>", on_select)
    
    if (last_processed_batch and last_processed_batch["batch_num"] != snapshot_last_plotted_num) or force_redraw[0]:
        force_redraw[0] = False
        
        if last_processed_batch:
            snapshot_last_plotted_num = last_processed_batch["batch_num"]
            
          
            prediction_text = last_processed_batch.get("prediction", "N/A")
            prediction_var.set(f"Prediction: {prediction_text}")
           
            
            log_label = f"Experimental Batch {last_processed_batch['batch_num']}"
            if not any(d['label'] == log_label for d in render_log):
                log_entry = {'type': 'exp', 'id': last_processed_batch["batch_index"], 'label': log_label}
                render_log.append(log_entry)
                log_listbox.insert(tk.END, log_label)

            current_channels = channel_selector['values']
            new_channel_count = last_processed_batch["n_channels"]
            if len(current_channels) != new_channel_count:
                channel_list = [str(i + 1) for i in range(new_channel_count)]
                channel_selector['values'] = channel_list
                if not exp_channel_var.get() and channel_list:
                    exp_channel_var.set(channel_list[0])

            if exp_channel_var.get():
                try:
                    selected_channel_idx = int(exp_channel_var.get()) - 1
                    denoised_channel_data = last_processed_batch["denoised_batch"][selected_channel_idx]
                    time_axis = last_processed_batch["time"]
                    batch_num = last_processed_batch["batch_num"]

                    ax.clear()
                    ax.plot(time_axis, denoised_channel_data)
                    ax.set_title(f"Latest Batch (#{batch_num}) - Denoised Channel {selected_channel_idx + 1}")
                    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.grid(True)
                    ax.autoscale(enable=True, axis='y', tight=True)
                    canvas.draw()
                except (ValueError, IndexError):
                    pass

    root.after(200, update_snapshot_waveform_plot, root, ax, canvas, channel_selector)

# -----------------------------
# Static Plotting Functions for Log Clicks
# -----------------------------
def show_static_waterfall(batch_index):
    if batch_index >= len(exp_signal_batches): return
    batch = exp_signal_batches[batch_index]
    n_channels, total_samples = batch.shape
    time_axis = np.linspace(0, total_samples / 1000, total_samples)
    normalized_batch = batch.copy()
    for i in range(n_channels):
        mean, std = np.mean(normalized_batch[i]), np.std(normalized_batch[i])
        if std > 1e-8:
            normalized_batch[i] = (normalized_batch[i] - mean) / std
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(normalized_batch, aspect='auto', cmap='viridis', origin='lower',
                   extent=[time_axis.min(), time_axis.max(), 1, n_channels])
    fig.colorbar(im, ax=ax, label='Normalized Amplitude')
    ax.set_title(f"Waterfall for Batch #{batch_index + 1}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Sensor Index")
    plt.tight_layout(); plt.show()

def show_static_denoised_waveform(batch_index):
    if batch_index >= len(exp_signal_batches): return
    
    win = tk.Toplevel(root)
    win.title(f"Denoised Waveforms for Batch #{batch_index + 1}")
    win.geometry("800x600")

    batch = exp_signal_batches[batch_index]
    denoised_batch = np.array([denoise_signal(ch) for ch in batch])
    
    n_channels, total_samples = batch.shape
    time_axis = np.linspace(0, total_samples / 1000, total_samples)
    
    top_frame = ttk.Frame(win)
    top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
    
    ttk.Label(top_frame, text="View Denoised Channel:").pack(side=tk.LEFT)
    channel_var = tk.StringVar()
    channel_selector = ttk.Combobox(top_frame, textvariable=channel_var, state='readonly', width=5)
    
    channel_list = [str(i + 1) for i in range(n_channels)]
    channel_selector['values'] = channel_list
    if channel_list:
        channel_var.set(channel_list[0])
    channel_selector.pack(side=tk.LEFT, padx=5)

    fig = Figure(dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_plot(event=None):
        selected_idx = int(channel_var.get()) - 1
        signal = denoised_batch[selected_idx]
        ax.clear()
        ax.plot(time_axis, signal)
        ax.set_title(f"Denoised Channel {selected_idx + 1}")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.grid(True)
        ax.autoscale(enable=True, axis='y', tight=True)
        canvas.draw()

    channel_selector.bind("<<ComboboxSelected>>", update_plot)
    update_plot()

# -----------------------------
# Log Select Function
# -----------------------------
def on_log_select(event):
    selection = event.widget.curselection()
    if not selection: return
    
    selected_log_index = selection[0]
    log_item = render_log[selected_log_index]
    
    if log_item['type'] == 'sim':
        sim_pipeline_index = log_item['id']
        render_waveform(auto=True, pipeline_idx_override=sim_pipeline_index)
        render_3d_pipeline(auto=True, pipeline_idx_override=sim_pipeline_index)
    elif log_item['type'] == 'exp':
        exp_batch_index = log_item['id']
        show_static_waterfall(exp_batch_index)
        show_static_denoised_waveform(exp_batch_index)

# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("CNN Pipeline Classifier")
root.geometry("850x700")
tab_control = ttk.Notebook(root)

# --- Global TK Variables ---
pipeline_var = tk.StringVar(value="1"); sensor_var = tk.StringVar(value="1")
exp_channel_var = tk.StringVar()
prediction_var = tk.StringVar(value="Prediction: N/A") # For the new prediction label
interpolation_enabled = tk.BooleanVar(value=True); max_time_var = tk.StringVar(value="2.5")
switch_axes_var = tk.BooleanVar(value=False)

# --- Simulation & Log Tabs ---
sim_tab = ttk.Frame(tab_control); tab_control.add(sim_tab, text='Simulation Data')
ttk.Button(sim_tab, text="Load .mat File", command=load_mat_file).pack(pady=10)
ttk.Label(sim_tab, text="Pipeline (1-N):").pack(); ttk.Entry(sim_tab, textvariable=pipeline_var).pack()
ttk.Label(sim_tab, text="Sensor (1-N):").pack(); ttk.Entry(sim_tab, textvariable=sensor_var).pack()
ttk.Button(sim_tab, text="Plot Waveform", command=lambda: render_waveform(auto=False)).pack(pady=5)
ttk.Button(sim_tab, text="Generate Confusion Matrix", command=show_confusion_matrix).pack(pady=5)
ttk.Button(sim_tab, text="Render 3D Pipeline", command=lambda: render_3d_pipeline(auto=False)).pack(pady=5)
ttk.Button(sim_tab, text="Predict Selected Pipeline", command=predict_simulation_pipeline).pack(pady=5)

log_tab = ttk.Frame(tab_control); tab_control.add(log_tab, text='Log')
ttk.Label(log_tab, text="Analysis History:").pack(pady=(10,0))
log_frame = ttk.Frame(log_tab)
log_frame.pack(expand=True, fill="both", padx=10, pady=10)
log_scrollbar = ttk.Scrollbar(log_frame)
log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
log_listbox = tk.Listbox(log_frame, yscrollcommand=log_scrollbar.set)
log_listbox.pack(side=tk.LEFT, expand=True, fill="both")
log_scrollbar.config(command=log_listbox.yview)
log_listbox.bind('<<ListboxSelect>>', on_log_select)

# --- Experimental Data Tab ---
exp_tab = ttk.Frame(tab_control); tab_control.add(exp_tab, text="Experimental Data")
exp_actions_frame = ttk.Frame(exp_tab)
exp_actions_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
ttk.Label(exp_actions_frame, text="View Denoised Channel:").pack(side=tk.LEFT, padx=(0, 5))
exp_channel_selector = ttk.Combobox(exp_actions_frame, textvariable=exp_channel_var, state='readonly', width=5)
exp_channel_selector.pack(side=tk.LEFT, padx=5)
ttk.Button(exp_actions_frame, text="Open 2D Pipeline Viewer", command=open_2d_pipeline_viewer).pack(side=tk.LEFT, padx=5)
ttk.Button(exp_actions_frame, text="Predict from Experimental Batch", command=predict_exp_data).pack(side=tk.LEFT, padx=5)

# Add the prediction label
prediction_label = ttk.Label(exp_tab, textvariable=prediction_var, font=("Helvetica", 14, "bold"))
prediction_label.pack(side=tk.TOP, pady=(5,0))

fig_exp = Figure(dpi=100)
ax_exp = fig_exp.add_subplot(111)
ax_exp.set_title("Waiting for first data batch...")
ax_exp.set_xlabel("Time (s)"); ax_exp.set_ylabel("Amplitude"); ax_exp.grid(True)
canvas_exp = FigureCanvasTkAgg(fig_exp, master=exp_tab)
canvas_exp.draw()
canvas_exp.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# --- Live Waterfall Tab ---
waterfall_tab = ttk.Frame(tab_control); tab_control.add(waterfall_tab, text="Live Waterfall")
controls_frame = ttk.Frame(waterfall_tab)
controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
ttk.Checkbutton(controls_frame, text="Enable Interpolation", variable=interpolation_enabled).pack(side=tk.LEFT, padx=5)
ttk.Checkbutton(controls_frame, text="Switch X/Y Axes", variable=switch_axes_var).pack(side=tk.LEFT, padx=5)
ttk.Label(controls_frame, text="Max Time (s):").pack(side=tk.LEFT, padx=(10, 2))
ttk.Entry(controls_frame, textvariable=max_time_var, width=8).pack(side=tk.LEFT)
fig_live = Figure(dpi=100)
ax_live = fig_live.add_subplot(111)
ax_live.set_title("Waiting for data..."); ax_live.set_xlabel("Time (s)"); ax_live.set_ylabel("Sensor Index")
im_live = ax_live.imshow(np.zeros((10, 10)), aspect='auto', cmap='viridis', origin='lower')
fig_live.colorbar(im_live, ax=ax_live, label='Amplitude'); fig_live.tight_layout()
canvas_live = FigureCanvasTkAgg(fig_live, master=waterfall_tab)
canvas_live.draw()
canvas_live.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# --- Finalize ---
tab_control.pack(expand=1, fill="both")
threading.Thread(target=watch_input_folder, daemon=True).start()
threading.Thread(target=watch_experimental_folder, daemon=True).start()
threading.Thread(target=process_data_for_plot, daemon=True).start()
root.after(200, update_waterfall_plot, root, ax_live, canvas_live, im_live)
root.after(200, update_snapshot_waveform_plot, root, ax_exp, canvas_exp, exp_channel_selector)
root.mainloop()