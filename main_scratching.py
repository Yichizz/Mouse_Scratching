# Example usage
from utils.scratch_detector import ScratchDetector

video_path = '../Scratching_Mouse/videos/I-5.mp4' # B-2, C-6, I-5 all right
pretrained_weights = 'runs/pose/fine_tune_11videos/weights/best.pt'
save_path = './I-5_.csv'

def main(video_path, pretrained_weights, save_path):
    detector = ScratchDetector(video_path, pretrained_weights, save_path, plot_prediction = True, high_conf_class= True)
    detector.process_video()

if __name__ == '__main__':
    main(video_path, pretrained_weights, save_path)
    