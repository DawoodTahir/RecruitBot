import { useEffect, useRef, useState } from "react";
import { X, Camera, RefreshCw } from "lucide-react";

interface CameraModalProps {
    onCapture: (file: File) => void;
    onClose: () => void;
}

const CameraModal = ({ onCapture, onClose }: CameraModalProps) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        startCamera();
        return () => stopCamera();
    }, []);

    const startCamera = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user" },
                audio: false,
            });
            setStream(mediaStream);
            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
            }
        } catch (err: any) {
            console.error("Camera access error:", err);
            let errorMessage = "Unable to access camera. Please check permissions.";
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                errorMessage = "Camera permission denied. Please allow access in your browser settings.";
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                errorMessage = "No camera found. Please ensure your device is connected.";
            } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
                errorMessage = "Camera is in use by another application.";
            } else if (err.message) {
                errorMessage = `Camera error: ${err.message}`;
            }
            setError(errorMessage);
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            setStream(null);
        }
    };

    const handleCapture = () => {
        if (videoRef.current && canvasRef.current) {
            const video = videoRef.current;
            const canvas = canvasRef.current;

            // Set canvas dimensions to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const context = canvas.getContext("2d");
            if (context) {
                // Flip horizontally for mirror effect if needed, but standard capture usually expects normal
                // context.scale(-1, 1); 
                // context.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);

                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob((blob) => {
                    if (blob) {
                        const file = new File([blob], `capture-${Date.now()}.png`, { type: "image/png" });
                        onCapture(file);
                        onClose(); // Close modal after capture
                    }
                }, "image/png");
            }
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in">
            <div className="relative w-full max-w-2xl overflow-hidden rounded-3xl border border-white/10 bg-slate-900 shadow-2xl mx-4">
                {/* Header */}
                <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
                    <h3 className="text-lg font-semibold text-white">Take Photo</h3>
                    <button
                        onClick={onClose}
                        className="rounded-full p-2 text-white/60 hover:bg-white/10 hover:text-white transition-colors"
                    >
                        <X className="size-5" />
                    </button>
                </div>

                {/* Video Feed */}
                <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
                    {error ? (
                        <div className="text-center p-6 text-rose-400">
                            <p>{error}</p>
                            <button
                                onClick={() => { setError(null); startCamera(); }}
                                className="mt-4 px-4 py-2 bg-white/10 rounded-lg text-sm text-white hover:bg-white/20 transition-colors"
                            >
                                Retry
                            </button>
                        </div>
                    ) : (
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className="h-full w-full object-cover transform scale-x-[-1]" // Mirror effect for user view
                        />
                    )}

                    <canvas ref={canvasRef} className="hidden" />
                </div>

                {/* Controls */}
                <div className="flex items-center justify-center gap-4 bg-slate-900/50 p-6 backdrop-blur-md">
                    <button
                        onClick={handleCapture}
                        disabled={!!error || !stream}
                        className="group relative flex h-16 w-16 items-center justify-center rounded-full border-4 border-white/20 bg-white/10 transition-all hover:border-[#40E0D0] hover:bg-[#40E0D0]/20 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <div className="h-12 w-12 rounded-full bg-white transition-transform group-hover:scale-90 group-active:scale-75" />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default CameraModal;
