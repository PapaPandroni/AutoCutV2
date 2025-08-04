"""
GUI Module for AutoCut

Provides a simple Tkinter-based interface for non-technical users
to create beat-synced highlight videos.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Callable
import threading
import os


class AutoCutGUI:
    """Main GUI application for AutoCut."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AutoCut - Beat-Synced Video Creator")
        self.root.geometry("600x500")
        
        # File paths
        self.video_files: List[str] = []
        self.music_file: Optional[str] = None
        self.output_path: Optional[str] = None
        
        # Processing state
        self.is_processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="AutoCut Video Generator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Video files section
        ttk.Label(main_frame, text="Video Files:").grid(row=1, column=0, sticky=tk.W)
        
        self.video_listbox = tk.Listbox(main_frame, height=6)
        self.video_listbox.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 10))
        
        video_button_frame = ttk.Frame(main_frame)
        video_button_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Button(video_button_frame, text="Add Videos", 
                  command=self.select_videos).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(video_button_frame, text="Remove Selected", 
                  command=self.remove_video).pack(side=tk.LEFT)
        
        # Music file section
        ttk.Label(main_frame, text="Music File:").grid(row=4, column=0, sticky=tk.W, pady=(20, 5))
        
        self.music_var = tk.StringVar(value="No music file selected")
        ttk.Label(main_frame, textvariable=self.music_var, 
                 relief=tk.SUNKEN, padding=5).grid(row=5, column=0, columnspan=2, 
                                                  sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(main_frame, text="Select Music", 
                  command=self.select_music).grid(row=6, column=0, sticky=tk.W)
        
        # Output location section
        ttk.Label(main_frame, text="Output Location:").grid(row=7, column=0, sticky=tk.W, pady=(20, 5))
        
        self.output_var = tk.StringVar(value="No output location selected")
        ttk.Label(main_frame, textvariable=self.output_var, 
                 relief=tk.SUNKEN, padding=5).grid(row=8, column=0, columnspan=2, 
                                                  sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(main_frame, text="Choose Output", 
                  command=self.select_output).grid(row=9, column=0, sticky=tk.W)
        
        # Settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), 
                           pady=(20, 10))
        
        # Tempo preference
        ttk.Label(settings_frame, text="Cutting Style:").grid(row=0, column=0, sticky=tk.W)
        self.tempo_var = tk.StringVar(value="balanced")
        tempo_combo = ttk.Combobox(settings_frame, textvariable=self.tempo_var,
                                  values=["energetic", "balanced", "buildup", "dramatic"],
                                  state="readonly", width=15)
        tempo_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Face priority
        self.face_priority_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Prioritize faces in videos", 
                       variable=self.face_priority_var).grid(row=1, column=0, columnspan=2, 
                                                            sticky=tk.W, pady=(10, 0))
        
        # Generate button and progress
        generate_frame = ttk.Frame(main_frame)
        generate_frame.grid(row=11, column=0, columnspan=2, pady=(20, 0))
        
        self.generate_button = ttk.Button(generate_frame, text="Generate Video", 
                                         command=self.generate_video, style="Accent.TButton")
        self.generate_button.pack()
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=12, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=13, column=0, columnspan=2, 
                                                               pady=(5, 0))
        
        # Configure column weights for resizing
        main_frame.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def select_videos(self):
        """Open file dialog to select video files."""
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        for file in files:
            if file not in self.video_files:
                self.video_files.append(file)
                self.video_listbox.insert(tk.END, os.path.basename(file))
                
    def remove_video(self):
        """Remove selected video from list."""
        selection = self.video_listbox.curselection()
        if selection:
            index = selection[0]
            self.video_listbox.delete(index)
            del self.video_files[index]
            
    def select_music(self):
        """Open file dialog to select music file."""
        file = filedialog.askopenfilename(
            title="Select Music File",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.aac"),
                ("All files", "*.*")
            ]
        )
        
        if file:
            self.music_file = file
            self.music_var.set(os.path.basename(file))
            
    def select_output(self):
        """Open file dialog to select output location."""
        file = filedialog.asksaveasfilename(
            title="Save Video As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if file:
            self.output_path = file
            self.output_var.set(os.path.basename(file))
            
    def validate_inputs(self) -> bool:
        """Validate that all required inputs are provided."""
        if not self.video_files:
            messagebox.showerror("Error", "Please select at least one video file.")
            return False
            
        if not self.music_file:
            messagebox.showerror("Error", "Please select a music file.")
            return False
            
        if not self.output_path:
            messagebox.showerror("Error", "Please choose an output location.")
            return False
            
        return True
        
    def update_progress(self, value: float, status: str = ""):
        """Update progress bar and status."""
        self.progress_var.set(value)
        if status:
            self.status_var.set(status)
        self.root.update_idletasks()
        
    def generate_video_thread(self):
        """Process video generation in separate thread."""
        try:
            self.update_progress(10, "Analyzing videos...")
            
            # TODO: Call actual processing functions
            # from .clip_assembler import assemble_clips
            # result = assemble_clips(
            #     self.video_files,
            #     self.music_file,
            #     self.output_path,
            #     self.tempo_var.get(),
            #     self.update_progress
            # )
            
            # Placeholder processing simulation
            import time
            for i in range(10, 101, 10):
                time.sleep(0.5)  # Simulate processing
                self.update_progress(i, f"Processing... {i}%")
                
            self.update_progress(100, "Complete!")
            messagebox.showinfo("Success", f"Video saved to: {self.output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.update_progress(0, "Error occurred")
        finally:
            self.is_processing = False
            self.generate_button.config(state="normal")
            
    def generate_video(self):
        """Start video generation process."""
        if not self.validate_inputs():
            return
            
        if self.is_processing:
            return
            
        self.is_processing = True
        self.generate_button.config(state="disabled")
        self.update_progress(0, "Starting...")
        
        # Start processing in separate thread
        thread = threading.Thread(target=self.generate_video_thread)
        thread.daemon = True
        thread.start()


def main():
    """Run the AutoCut GUI application."""
    root = tk.Tk()
    app = AutoCutGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()