# Example usage
from utils.scratch_detector import ScratchDetector

video_path = '../Scratching_Mouse/videos/F-3.mp4'
pretrained_weights = 'runs/pose/fine_tune_11videos/weights/best.pt'
save_path = './F-3_.csv'

def main(video_path, pretrained_weights, save_path):
    detector = ScratchDetector(video_path, pretrained_weights, save_path, plot_prediction = True, high_conf_class= False)
    detector.process_video()

if __name__ == '__main__':
    main(video_path, pretrained_weights, save_path)
    