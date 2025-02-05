import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageLabelingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("图片标注工具")

        # 初始化变量
        self.image_dir = None
        self.image_files = []
        self.current_index = 0
        self.output_file = None
        self.labels_data = {}
        self.current_category = None
        self.current_gender = None
        self.selected_category_button = None
        self.selected_gender_button = None
        self.is_bug = False

        # 定义男女分类选项
        self.male_categories = ["1-4", "5-7", "8-10", "11-12", "13-15", "16-19", "20-24", "25-30", "35-40"]
        self.female_categories = ["12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-35", "36-40", "50+"]

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 获取屏幕分辨率
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_width}x{screen_height}")

        self.select_dir_btn = tk.Button(self.master, text="选择图片目录", command=self.select_directory)
        self.select_dir_btn.pack()

        # 主图片显示区域
        self.image_canvas = tk.Canvas(self.master, width=500, height=500, bg="gray")
        self.image_canvas.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

        # Bug按钮（放在显眼的位置）
        self.bug_button = tk.Button(self.master, 
                                  text="BUG图片", 
                                  command=self.mark_as_bug,
                                  height=2,
                                  width=20,
                                  bg="red",
                                  fg="black",
                                  font=("Helvetica", 12, "bold"))
        self.bug_button.pack(pady=10)

        # 参考图片显示
        self.setup_reference_images()

        # 图片文件名显示标签
        self.image_label = tk.Label(self.master, text="请加载图片目录")
        self.image_label.pack()

        # 分类按钮框架
        self.label_buttons_frame = tk.Frame(self.master)
        self.label_buttons_frame.pack()

        # 性别按钮框架
        self.gender_buttons_frame = tk.Frame(self.master)
        self.gender_buttons_frame.pack()

        # 状态显示框架
        self.status_frame = tk.Frame(self.master)
        self.status_frame.pack()
        self.status_label = tk.Label(self.status_frame, text="当前分类：未选择 | 当前性别：未选择")
        self.status_label.pack()

        # 添加进度按钮
        self.progress_btn = tk.Button(self.master, text="显示进度", command=self.show_progress)
        self.progress_btn.pack(side=tk.LEFT, padx=10)

        # 导航按钮
        self.prev_btn = tk.Button(self.master, text="上一张", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT)

    def setup_reference_images(self):
        # 右上方参考图
        self.image_canvas2 = tk.Canvas(self.master, width=350, height=370, bg="gray")
        self.image_canvas2.place(relx=0.87, rely=0.21, anchor=tk.CENTER)

        img_female = Image.open("/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/bodyfatstandard/female.jpg")
        img_female = img_female.resize((350, 350))
        self.tk_image_female = ImageTk.PhotoImage(img_female)
        self.image_canvas2.create_image(175, 200, image=self.tk_image_female)

        # 左上方参考图
        self.image_canvas3 = tk.Canvas(self.master, width=350, height=370, bg="gray")
        self.image_canvas3.place(relx=0.13, rely=0.21, anchor=tk.CENTER)

        img_male = Image.open("/Users/yuntianzeng/Desktop/ML/Body-Fat-Regression-from-Reddit-Image-Dataset/bodyfatstandard/male.jpg")
        img_male = img_male.resize((350, 350))
        self.tk_image_male = ImageTk.PhotoImage(img_male)
        self.image_canvas3.create_image(175, 200, image=self.tk_image_male)

    def select_directory(self):
        self.image_dir = filedialog.askdirectory()
        if not self.image_dir:
            messagebox.showwarning("警告", "未选择任何目录！")
            return

        # 获取文件夹名作为JSON文件名
        folder_name = os.path.basename(self.image_dir)
        self.output_file = f"{folder_name}.json"
        
        # 获取所有图片文件
        all_images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not all_images:
            messagebox.showerror("错误", "选定目录中没有图片文件！")
            return

        # 加载已有的标注数据
        self.load_existing_labels()
        
        # 过滤掉已标注的图片
        self.image_files = [img for img in all_images if img not in self.labels_data]
        
        if not self.image_files:
            messagebox.showinfo("完成", "所有图片都已标注完成！")
            return
        
        # 显示进度信息
        total_images = len(all_images)
        labeled_images = len(self.labels_data)
        remaining_images = len(self.image_files)
        
        messagebox.showinfo("标注进度", 
                           f"总共有 {total_images} 张图片\n"
                           f"已标注 {labeled_images} 张\n"
                           f"剩余 {remaining_images} 张待标注")
        
        self.current_index = 0
        self.show_image()

    def load_existing_labels(self):
        """加载已有的标注数据"""
        self.labels_data = {}
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as jsonfile:
                    for line in jsonfile:
                        try:
                            record = json.loads(line.strip())
                            if "image" in record:
                                self.labels_data[record["image"]] = record
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                messagebox.showerror("错误", f"读取标注文件时出错：{str(e)}")
                self.labels_data = {}

    def show_progress(self):
        """显示当前标注进度"""
        if hasattr(self, 'image_dir') and self.image_dir:
            all_images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            total_images = len(all_images)
            labeled_images = len(self.labels_data)
            remaining_images = len(self.image_files)
            
            messagebox.showinfo("标注进度", 
                               f"总共有 {total_images} 张图片\n"
                               f"已标注 {labeled_images} 张\n"
                               f"剩余 {remaining_images} 张待标注")

    def mark_as_bug(self):
        """标记当前图片为bug并自动进入下一张"""
        # 先检查是否已选择目录且有图片
        if not self.image_dir or not self.image_files:
            messagebox.showwarning("警告", "请先选择包含图片的目录！")
            return
        
        self.is_bug = True
        self.current_category = "bug"
        self.current_gender = None
        self.label_image()
        self.is_bug = False  # 重置bug状态

    def update_label_buttons(self):
        """根据性别更新分类按钮"""
        # 清空现有按钮
        for widget in self.label_buttons_frame.winfo_children():
            widget.destroy()
        self.selected_category_button = None

        # 清空性别按钮
        for widget in self.gender_buttons_frame.winfo_children():
            widget.destroy()
        self.selected_gender_button = None

        # 创建性别按钮
        genders = ["男", "女"]
        for gender in genders:
            btn = tk.Button(self.gender_buttons_frame, text=gender, command=lambda g=gender: self.set_gender(g))
            btn.pack(side=tk.LEFT)

        self.update_status_label()

    def set_gender(self, gender):
        """设置性别并更新分类按钮"""
        self.current_gender = gender

        # 更新性别按钮外观
        if self.selected_gender_button and self.selected_gender_button.winfo_exists():
            self.selected_gender_button.config(relief="raised")

        for widget in self.gender_buttons_frame.winfo_children():
            if widget.cget("text") == gender:
                widget.config(relief="sunken")
                self.selected_gender_button = widget
                break

        # 根据性别显示对应的分类选项
        for widget in self.label_buttons_frame.winfo_children():
            widget.destroy()

        categories = self.male_categories if gender == "男" else self.female_categories
        for category in categories:
            btn = tk.Button(self.label_buttons_frame, text=category, command=lambda c=category: self.set_category(c))
            btn.pack(side=tk.LEFT)

        self.update_status_label()

    def set_category(self, category):
        self.current_category = category

        # 重置之前选中的按钮外观
        if self.selected_category_button and self.selected_category_button.winfo_exists():
            self.selected_category_button.config(relief="raised")

        # 设置当前按钮的外观
        for widget in self.label_buttons_frame.winfo_children():
            if widget.cget("text") == category:
                widget.config(relief="sunken")
                self.selected_category_button = widget
                break

        self.update_status_label()
        self.check_complete()

    def update_status_label(self):
        category_text = self.current_category if self.current_category else "未选择"
        gender_text = self.current_gender if self.current_gender else "未选择"
        self.status_label.config(text=f"当前分类：{category_text} | 当前性别：{gender_text}")

    def check_complete(self):
        """检查是否可以保存标注"""
        if self.is_bug or (self.current_category and self.current_gender):
            self.label_image()

    def label_image(self):
        """保存标注信息"""
        # 添加安全检查
        if not self.image_files:
            messagebox.showwarning("警告", "没有可标注的图片！")
            return
        
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            messagebox.showwarning("警告", "图片索引超出范围！")
            return
            
        image_name = self.image_files[self.current_index]
        
        if self.is_bug:
            self.labels_data[image_name] = {
                "category": "bug",
                "gender": None
            }
        else:
            self.labels_data[image_name] = {
                "category": self.current_category,
                "gender": self.current_gender
            }

        self.save_labels()

        self.current_category = None
        self.current_gender = None
        self.selected_category_button = None
        self.selected_gender_button = None

        self.next_image()

    def show_image(self):
        if 0 <= self.current_index < len(self.image_files):
            image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
            self.image_label.config(text=f"当前图片：{self.image_files[self.current_index]}")
            self.display_image(image_path)
            self.update_label_buttons()
        else:
            messagebox.showinfo("完成", "所有图片标注完成！")

    def display_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((500, 500))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_canvas.create_image(250, 250, image=self.tk_image)

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()
        else:
            messagebox.showwarning("提示", "已经是第一张图片了！")

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()
        else:
            messagebox.showinfo("完成", "所有图片标注完成！")

    def save_labels(self):
        """保存标注到JSON文件"""
        try:
            with open(self.output_file, "w", encoding="utf-8") as jsonfile:
                for image_name, data in self.labels_data.items():
                    json.dump({"image": image_name, **data}, jsonfile, ensure_ascii=False)
                    jsonfile.write("\n")
        except Exception as e:
            messagebox.showerror("错误", f"保存标注文件时出错：{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.mainloop()