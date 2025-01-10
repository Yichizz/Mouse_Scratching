# Example usage
from utils.scratch_detector import ScratchDetector

video_path = '../Scratching_Mouse/videos/A-1.mp4'
pretrained_weights = 'runs/pose/train/weights/best.pt'
save_path = './A-1.csv'

def main(video_path, pretrained_weights, save_path):
    detector = ScratchDetector(video_path, pretrained_weights, save_path, plot_prediction = False)
    detector.process_video()

if __name__ == '__main__':
    main(video_path, pretrained_weights, save_path)
    