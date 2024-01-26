import time
import csv
import threading
import pathlib
import numpy as np
from jtop import jtop

from typing import Optional, Callable, Literal


class ResultShower:
    def __init__(self):
        self.clear()

    def append(self, data):
        self.data.append(data)

    def clear(self):
        self.data = []

    def summary(self, show: bool, save: Optional[str] = None):
        print("Not implemented")

    def plot(self, show: bool, save: Optional[str] = None):
        print("Not implemented")

    def save_csv(self, save: str, header: Optional[list] = None):
        with open(save, "w") as f:
            writer = csv.writer(f)
            if header is not None:
                writer.writerow(header)
            for d in self.data:
                if isinstance(d, list):
                    writer.writerow(d)
                else:
                    writer.writerow([d])
            print(f"saved csv to {save}")


class TimeResultShower(ResultShower):
    def __init__(self):
        super().__init__()

    def summary(self, show: bool = True, save: Optional[str] = None):
        format_str = f"""
Time Result===============================
total loop={len(self.data)}
total inference time={sum(self.data)}
avg inference time={sum(self.data) / len(self.data)}
fps={len(self.data) / sum(self.data)}
==========================================
"""
        if show:
            print(format_str)
        if save is not None:
            with open(save, "w") as f:
                f.write(format_str)

    def plot(self, show: bool = True, save: Optional[str] = None):
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # axis x: loop, axis y: time (s)
        plt.figure(figsize=(10, 5))
        plt.title("Inference time")
        plt.xlabel("Loop")
        plt.ylabel("Time (ms)")
        plt.plot([d * 1000 for d in self.data])

        # draw average line
        avg = sum(self.data) * 1000 / len(self.data)
        plt.axhline(avg, color="r", linestyle="--", label="Average")
        plt.legend()

        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            print(f"saved figure to {save}")
        plt.close()

    def save_csv(self, save: str):
        header = ["time"]
        super().save_csv(save, header)


class PowerLogger(ResultShower):
    from jtop import jtop

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()

    def append(self, data):
        timestamp = time.time()
        self.data.append([timestamp, data])

    def measure_process(self, interval: float = 1.0):
        """
        Power measurement process
        It is recommended to run this process in a separate thread
        """
        with self.jtop(interval=interval) as jetson:
            while not self.stop_event.is_set():
                # stats = jetson.stats
                power = jetson.power
                # time = stats["time"] - self.start_time
                # time = time.total_seconds()
                # cpu_gpu = power["rail"]["VDD_CPU_GPU_CV"]["power"]
                # soc = power["rail"]["VDD_SOC"]["power"]
                total = power["tot"]["power"]
                # dla0 = False if stats["DLA0_CORE"] == "OFF" else True
                # dla1 = False if stats["DLA1_CORE"] == "OFF" else True
                # gpu = stats["GPU"]
                # data = [time, cpu_gpu, soc, total, dla0, dla1, gpu]
                self.append(total)
                time.sleep(interval)

    def start_measure(self, interval: float = 1.0):
        self.measure_thread = threading.Thread(
            target=self.measure_process, args=(interval,)
        )
        self.measure_thread.start()

    def stop_measure(self):
        self.stop_event.set()
        self.measure_thread.join()

    def summary(self, show: bool = True, save: Optional[str] = None):
        if len(self.data) == 0:
            print("No data")
            return
        elapsed = self.data[-1][0] - self.data[0][0]
        avg_power = sum([d[1] for d in self.data]) / len(self.data)
        peak_power = max([d[1] for d in self.data])
        min_power = min([d[1] for d in self.data])
        format_str = f"""
Power Result==============================
elapsed time={elapsed}
avg power={avg_power}
peak power={peak_power}
min power={min_power}
==========================================
"""
        if show:
            print(format_str)
        if save is not None:
            with open(save, "w") as f:
                f.write(format_str)

    def plot(self, show: bool = True, save: Optional[str] = None):
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # time to relative time
        start_time = self.data[0][0]
        df = {"time": [], "total": []}
        for d in self.data:
            df["time"].append(d[0] - start_time)
            df["total"].append(d[1])

        _, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(df["time"], df["total"], label="Total")
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (mW)")
        ax.set_title("Power consumption")
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            print(f"saved figure to {save}")
        plt.close()

    def save_csv(self, save: str):
        header = ["time", "total"]
        super().save_csv(save, header)


def sarnn_image_postprocess(image: np.ndarray) -> np.ndarray:
    image = image.reshape(3, 128, 128)
    image = image.transpose(1, 2, 0)
    image[np.where(image < 0.0)] = 0.0
    image[np.where(image > 1.0)] = 1.0
    image = image * 255.0
    image = image.astype(np.uint8)
    return image


class InferenceResultShower(ResultShower):
    def __init__(
        self,
        model: Literal["sarnn", "cnnrnn", "cnnrnnln", "caebn"] = "sarnn",
        image_postprocess: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        joint_postprocess: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    ):
        super().__init__()
        self.model = model
        self.image_postprocess = image_postprocess
        self.joint_postprocess = joint_postprocess

    def append(
        self,
        input_image: np.ndarray,
        pred_image: np.ndarray,
        input_joint: np.ndarray,
        pred_joint: np.ndarray,
        enc_pts: np.ndarray,
        dec_pts: np.ndarray,
        elapsed_time: float,
    ):
        now = time.time()
        pred_image = self.image_postprocess(pred_image.copy())
        input_image = self.image_postprocess(input_image.copy())
        pred_joint = self.joint_postprocess(pred_joint.copy())
        input_joint = self.joint_postprocess(input_joint.copy())
        self.data.append(
            [
                now,
                pred_image,
                pred_joint,
                enc_pts.copy(),
                dec_pts.copy(),
                input_image,
                input_joint,
                elapsed_time,
            ]
        )

    def summary(self, show: bool = True, save: Optional[str] = None):
        import sklearn.metrics as metrics

        jointmses = []
        if self.model in ["sarnn", "cnnrnn", "cnnrnnln"]:
            for i in range(len(self.data) - 1):
                input_joint = self.data[i + 1][6]
                pred_joint = self.data[i][2]
                mse = metrics.mean_squared_error(input_joint, pred_joint)
                jointmses.append(mse)
            format_str = f"""
Inference Result==========================
total loop={len(self.data)}
MSE of joint angle={sum(jointmses) / len(jointmses)}
MAX MSE of joint angle={max(jointmses)}
MIN MSE of joint angle={min(jointmses)}
==========================================
"""
        else:
            format_str = f"""
Inference Result==========================
total loop={len(self.data)}
MSE of joint angle=---
==========================================
"""
        if show:
            print(format_str)
        if save is not None:
            with open(save, "w") as f:
                f.write(format_str)

    def save_numpy(self, image_save: str, joint_save: str):
        np.save(image_save, np.array([d[1] for d in self.data]))
        np.save(joint_save, np.array([d[2] for d in self.data]))

    def plot(self, show: bool = True, save: Optional[str] = None):
        import matplotlib
        import sklearn.metrics as metrics

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as anim

        fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=60)

        img_size = 128
        k_dim = 5
        if self.model == "sarnn":
            enc_pts = np.array([d[3] for d in self.data])
            dec_pts = np.array([d[4] for d in self.data])
            enc_pts = enc_pts.reshape(-1, k_dim, 2) * img_size
            dec_pts = dec_pts.reshape(-1, k_dim, 2) * img_size
            enc_pts = np.clip(enc_pts, 0, img_size)
            dec_pts = np.clip(dec_pts, 0, img_size)

        if self.model in ["sarnn", "cnnrnn", "cnnrnnln"]:
            pred_joint = np.array([d[2] for d in self.data])
            input_joint = np.array([d[6] for d in self.data])

        def anim_update(i):
            _, pred_image, _, _, _, input_image, _, elapsed_time = self.data[i]
            for j in range(3):
                ax[j].cla()

            # plot camera image
            ax[0].imshow(input_image)
            if self.model == "sarnn":
                for j in range(k_dim):
                    ax[0].plot(
                        enc_pts[i, j, 0], enc_pts[i, j, 1], "bo", markersize=6
                    )  # encoder
                    ax[0].plot(
                        dec_pts[i, j, 0],
                        dec_pts[i, j, 1],
                        "rx",
                        markersize=6,
                        markeredgewidth=2,
                    )  # decoder
            ax[0].axis("off")
            ax[0].set_title("Input image")

            # plot predicted image
            ax[1].imshow(pred_image)
            ax[1].axis("off")
            ax[1].set_title(
                f"Predicted image \nelapsed time={(elapsed_time * 1000):.2f} ms"
            )

            # plot joint angle
            if self.model in ["sarnn", "cnnrnn", "cnnrnnln"]:
                ax[2].set_ylim(-1.0, 2.0)
                ax[2].set_xlim(0, T)
                ax[2].plot(input_joint[1:], linestyle="dashed", c="k")
                for joint_idx in range(8):
                    ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
                ax[2].set_xlabel("Step")
                if i < T - 1:
                    # print mse
                    # input[t+1] is the answer for pred[t]
                    mse = metrics.mean_squared_error(
                        input_joint[i + 1],
                        pred_joint[i],
                    )
                    # show x.xxe-xx
                    ax[2].set_title(f"Joint angles\nmse={mse:.2e}")
                else:
                    ax[2].set_title(f"Joint angles\nmse=---")

        T = len(self.data)
        ani = anim.FuncAnimation(fig, anim_update, interval=100, frames=T)
        if show:
            plt.show()
        if save is not None:
            if pathlib.Path(save).suffix != ".mp4":
                save += ".mp4"
            print("saving animation, this may take a while...")
            ani.save(save)
            print(f"saved animation to {save}")


if __name__ == "__main__":
    # test
    test_time = [0.1, 0.2, 0.3, 0.2, 0.5]
    tr = TimeResultShower()
    tr.data = test_time
    tr.summary()
    tr.plot(show=False, save="test1.png")
    tr.save_csv("test1.csv")

    pl = PowerLogger()
    pl.start_measure(interval=0.1)
    print("wait 5 sec")
    time.sleep(5)
    pl.stop_measure()
    pl.summary()
    pl.plot(show=False, save="test2.png")
    pl.save_csv("test2.csv")
