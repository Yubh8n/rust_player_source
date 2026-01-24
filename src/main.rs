use iced::widget::{button, canvas, column, container, image, row, text, Space, vertical_space};
use iced::{alignment, mouse, Element, Length, Rectangle, Renderer, Subscription, Task, Theme};
use iced::theme::Palette;
use iced::{Border, Color, Shadow};
use iced::widget::canvas::{Cache, Geometry, Path, Program};
use rodio::{OutputStream, Sink, Source};
use rustfft::{FftPlanner, num_complex::Complex};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::collections::VecDeque;

const LOGO_BYTES: &[u8] = include_bytes!("../top1_logo.png");
const SMILEY_BYTES: &[u8] = include_bytes!("../smiley.png");

const FFT_SIZE: usize = 2048;
const NUM_BARS: usize = 32;

fn main() -> iced::Result {
    iced::application("PartyFM", PartyFMPlayer::update, PartyFMPlayer::view)
        .subscription(PartyFMPlayer::subscription)
        .theme(PartyFMPlayer::theme)
        .window_size((420.0, 420.0))
        .resizable(false)
        .antialiasing(true)
        .run()
}

#[derive(Debug, Clone)]
enum Message {
    TogglePlay,
    VolumeUp,
    VolumeDown,
    Tick,
}

#[derive(Debug, Clone, PartialEq)]
enum StreamState {
    Stopped,
    Connecting,
    Playing,
    Error(String),
}

impl Default for StreamState {
    fn default() -> Self {
        StreamState::Stopped
    }
}

enum AudioCommand {
    Play,
    Stop,
    VolumeUp,
    VolumeDown,
}

struct AudioController {
    tx: Sender<AudioCommand>,
}

struct StreamingBuffer {
    buffer: Arc<Mutex<VecDeque<u8>>>,
    stop_flag: Arc<Mutex<bool>>,
}

impl StreamingBuffer {
    fn new() -> (Self, Arc<Mutex<VecDeque<u8>>>, Arc<Mutex<bool>>) {
        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(500_000)));
        let stop_flag = Arc::new(Mutex::new(false));
        (
            Self {
                buffer: buffer.clone(),
                stop_flag: stop_flag.clone(),
            },
            buffer,
            stop_flag,
        )
    }
}

impl Read for StreamingBuffer {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            if *self.stop_flag.lock().unwrap() {
                return Ok(0);
            }

            {
                let mut buffer = self.buffer.lock().unwrap();
                if !buffer.is_empty() {
                    let to_read = buf.len().min(buffer.len());
                    for i in 0..to_read {
                        buf[i] = buffer.pop_front().unwrap();
                    }
                    return Ok(to_read);
                }
            }

            thread::sleep(Duration::from_millis(10));
        }
    }
}

struct Mp3StreamSource {
    decoder: minimp3::Decoder<StreamingBuffer>,
    current_frame: Vec<i16>,
    frame_index: usize,
    sample_rate: u32,
    channels: u16,
    fft_buffer: Arc<Mutex<Vec<f32>>>,
    sample_buffer: Vec<f32>,
}

impl Mp3StreamSource {
    fn new(reader: StreamingBuffer, fft_buffer: Arc<Mutex<Vec<f32>>>) -> Option<Self> {
        let mut decoder = minimp3::Decoder::new(reader);

        match decoder.next_frame() {
            Ok(frame) => Some(Self {
                decoder,
                sample_rate: frame.sample_rate as u32,
                channels: frame.channels as u16,
                current_frame: frame.data,
                frame_index: 0,
                fft_buffer,
                sample_buffer: Vec::with_capacity(FFT_SIZE),
            }),
            Err(_) => None,
        }
    }
}

impl Iterator for Mp3StreamSource {
    type Item = i16;

    fn next(&mut self) -> Option<Self::Item> {
        if self.frame_index < self.current_frame.len() {
            let sample = self.current_frame[self.frame_index];
            self.frame_index += 1;

            // Collect samples for FFT (mono - average if stereo)
            if self.channels == 2 && self.frame_index % 2 == 0 {
                let prev = self.current_frame.get(self.frame_index.saturating_sub(2)).copied().unwrap_or(0);
                let avg = ((prev as i32 + sample as i32) / 2) as f32 / 32768.0;
                self.sample_buffer.push(avg);
            } else if self.channels == 1 {
                self.sample_buffer.push(sample as f32 / 32768.0);
            }

            // Run FFT when buffer is full
            if self.sample_buffer.len() >= FFT_SIZE {
                self.run_fft();
            }

            return Some(sample);
        }

        match self.decoder.next_frame() {
            Ok(frame) => {
                self.current_frame = frame.data;
                self.frame_index = 0;
                if !self.current_frame.is_empty() {
                    let sample = self.current_frame[self.frame_index];
                    self.frame_index += 1;
                    Some(sample)
                } else {
                    None
                }
            }
            Err(minimp3::Error::Eof) => None,
            Err(_) => None,
        }
    }
}

impl Mp3StreamSource {
    fn run_fft(&mut self) {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        // Apply Hann window and convert to complex
        let mut buffer: Vec<Complex<f32>> = self.sample_buffer.iter().enumerate().map(|(i, &s)| {
            let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos());
            Complex::new(s * window, 0.0)
        }).collect();

        fft.process(&mut buffer);

        // Calculate magnitude for frequency bins
        let mut magnitudes = vec![0.0f32; NUM_BARS];
        let useful_bins = FFT_SIZE / 8; // Focus on lower frequencies

        for bar in 0..NUM_BARS {
            // Logarithmic frequency distribution
            let start = ((bar as f32 / NUM_BARS as f32).powf(1.8) * useful_bins as f32) as usize;
            let end = (((bar + 1) as f32 / NUM_BARS as f32).powf(1.8) * useful_bins as f32) as usize;
            let end = end.max(start + 1).min(useful_bins);

            let mut sum = 0.0;
            for i in start..end {
                // Normalize by FFT size
                let normalized = buffer[i].norm() / FFT_SIZE as f32;
                sum += normalized;
            }
            let avg = sum / (end - start) as f32;

            // Convert to dB-like scale for better visualization
            let db = if avg > 0.0001 {
                (20.0 * avg.log10() + 60.0) / 60.0  // Normalize to ~0-1 range
            } else {
                0.0
            };
            magnitudes[bar] = db.max(0.0).min(1.0);
        }

        // Update shared FFT buffer with smoothing
        if let Ok(mut fft_buf) = self.fft_buffer.lock() {
            if fft_buf.len() != NUM_BARS {
                *fft_buf = vec![0.0; NUM_BARS];
            }
            for i in 0..NUM_BARS {
                // Smooth: fast attack, slow decay
                if magnitudes[i] > fft_buf[i] {
                    fft_buf[i] = fft_buf[i] * 0.2 + magnitudes[i] * 0.8;
                } else {
                    fft_buf[i] = fft_buf[i] * 0.92 + magnitudes[i] * 0.08;
                }
            }
        }

        self.sample_buffer.clear();
    }
}

impl Source for Mp3StreamSource {
    fn current_frame_len(&self) -> Option<usize> { None }
    fn channels(&self) -> u16 { self.channels }
    fn sample_rate(&self) -> u32 { self.sample_rate }
    fn total_duration(&self) -> Option<Duration> { None }
}

impl AudioController {
    fn new(status_tx: Sender<StreamState>, now_playing: Arc<Mutex<String>>, fft_buffer: Arc<Mutex<Vec<f32>>>) -> Self {
        let (tx, rx): (Sender<AudioCommand>, Receiver<AudioCommand>) = mpsc::channel();

        thread::spawn(move || {
            let mut sink: Option<Sink> = None;
            let mut _stream: Option<OutputStream> = None;
            let mut stop_flag: Option<Arc<Mutex<bool>>> = None;
            let mut network_thread: Option<thread::JoinHandle<()>> = None;
            let mut volume_level: i32 = 10;

            loop {
                match rx.recv() {
                    Ok(AudioCommand::Play) => {
                        if let Some(flag) = stop_flag.take() {
                            *flag.lock().unwrap() = true;
                        }
                        sink = None;
                        _stream = None;
                        if let Some(handle) = network_thread.take() {
                            let _ = handle.join();
                        }

                        // Clear FFT buffer
                        if let Ok(mut buf) = fft_buffer.lock() {
                            buf.clear();
                        }

                        let _ = status_tx.send(StreamState::Connecting);

                        let (reader, buffer, flag) = StreamingBuffer::new();
                        stop_flag = Some(flag.clone());

                        let net_buffer = buffer.clone();
                        let net_flag = flag.clone();
                        let net_status = status_tx.clone();
                        let net_now_playing = now_playing.clone();

                        network_thread = Some(thread::spawn(move || {
                            if let Err(e) = stream_audio(net_buffer, net_flag.clone(), net_now_playing) {
                                if !*net_flag.lock().unwrap() {
                                    let _ = net_status.send(StreamState::Error(e));
                                }
                            }
                        }));

                        thread::sleep(Duration::from_millis(1500));

                        let has_data = {
                            let buf = buffer.lock().unwrap();
                            buf.len() > 10000
                        };

                        if !has_data {
                            *flag.lock().unwrap() = true;
                            let _ = status_tx.send(StreamState::Error("Ingen data".into()));
                            continue;
                        }

                        if let Ok((stream, stream_handle)) = OutputStream::try_default() {
                            if let Ok(new_sink) = Sink::try_new(&stream_handle) {
                                if let Some(source) = Mp3StreamSource::new(reader, fft_buffer.clone()) {
                                    new_sink.set_volume(volume_level as f32 / 10.0);
                                    new_sink.append(source);
                                    sink = Some(new_sink);
                                    _stream = Some(stream);
                                    let _ = status_tx.send(StreamState::Playing);
                                    continue;
                                }
                            }
                        }

                        *flag.lock().unwrap() = true;
                        let _ = status_tx.send(StreamState::Error("Audio fejl".into()));
                    }
                    Ok(AudioCommand::Stop) => {
                        if let Some(flag) = stop_flag.take() {
                            *flag.lock().unwrap() = true;
                        }
                        sink = None;
                        _stream = None;
                        // Clear FFT on stop
                        if let Ok(mut buf) = fft_buffer.lock() {
                            for v in buf.iter_mut() {
                                *v = 0.0;
                            }
                        }
                        let _ = status_tx.send(StreamState::Stopped);
                    }
                    Ok(AudioCommand::VolumeUp) => {
                        if volume_level < 10 {
                            volume_level += 1;
                        }
                        if let Some(ref s) = sink {
                            s.set_volume(volume_level as f32 / 10.0);
                        }
                    }
                    Ok(AudioCommand::VolumeDown) => {
                        if volume_level > 0 {
                            volume_level -= 1;
                        }
                        if let Some(ref s) = sink {
                            s.set_volume(volume_level as f32 / 10.0);
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Self { tx }
    }

    fn play(&self) { let _ = self.tx.send(AudioCommand::Play); }
    fn stop(&self) { let _ = self.tx.send(AudioCommand::Stop); }
    fn volume_up(&self) { let _ = self.tx.send(AudioCommand::VolumeUp); }
    fn volume_down(&self) { let _ = self.tx.send(AudioCommand::VolumeDown); }
}

fn stream_audio(buffer: Arc<Mutex<VecDeque<u8>>>, stop_flag: Arc<Mutex<bool>>, now_playing: Arc<Mutex<String>>) -> Result<(), String> {
    let mut stream = TcpStream::connect("stream1.partyfm.dk:80")
        .map_err(|e| format!("Ingen forbindelse: {}", e))?;

    stream.set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| format!("Timeout fejl: {}", e))?;

    let request = "GET /Party128web HTTP/1.1\r\nHost: stream1.partyfm.dk\r\nUser-Agent: PartyFM-Player/1.0\r\nAccept: */*\r\nIcy-MetaData: 1\r\nConnection: keep-alive\r\n\r\n";
    stream.write_all(request.as_bytes())
        .map_err(|e| format!("Send fejl: {}", e))?;

    let mut response = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut headers_done = false;
    let mut metaint: usize = 0;
    let mut bytes_until_meta: usize = 0;

    loop {
        if *stop_flag.lock().unwrap() {
            break;
        }

        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                if !headers_done {
                    response.extend_from_slice(&chunk[..n]);
                    if let Some(pos) = find_header_end(&response) {
                        headers_done = true;

                        let header_str = String::from_utf8_lossy(&response[..pos]);
                        for line in header_str.lines() {
                            if line.to_lowercase().starts_with("icy-metaint:") {
                                if let Ok(val) = line[12..].trim().parse::<usize>() {
                                    metaint = val;
                                    bytes_until_meta = val;
                                }
                            }
                        }

                        let body_start = pos + 4;
                        if body_start < response.len() {
                            let body = &response[body_start..];
                            process_stream_data(body, &buffer, &mut bytes_until_meta, metaint, &now_playing, &stop_flag, &mut stream)?;
                        }
                    }
                } else {
                    process_stream_data(&chunk[..n], &buffer, &mut bytes_until_meta, metaint, &now_playing, &stop_flag, &mut stream)?;
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                drop(stream);
                stream = TcpStream::connect("stream1.partyfm.dk:80")
                    .map_err(|e| format!("Reconnect fejl: {}", e))?;
                stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
                stream.write_all(request.as_bytes()).ok();
                headers_done = false;
                response.clear();
                metaint = 0;
                bytes_until_meta = 0;
            }
            Err(e) => return Err(format!("Læsefejl: {}", e)),
        }
    }

    Ok(())
}

fn process_stream_data(
    data: &[u8],
    buffer: &Arc<Mutex<VecDeque<u8>>>,
    bytes_until_meta: &mut usize,
    metaint: usize,
    now_playing: &Arc<Mutex<String>>,
    stop_flag: &Arc<Mutex<bool>>,
    stream: &mut TcpStream,
) -> Result<(), String> {
    let mut i = 0;

    while i < data.len() {
        if *stop_flag.lock().unwrap() {
            return Ok(());
        }

        if metaint > 0 && *bytes_until_meta == 0 {
            let meta_len = (data[i] as usize) * 16;
            i += 1;

            if meta_len > 0 {
                let mut meta_bytes = Vec::new();
                let available = (data.len() - i).min(meta_len);
                meta_bytes.extend_from_slice(&data[i..i + available]);
                i += available;

                while meta_bytes.len() < meta_len {
                    let mut extra = [0u8; 256];
                    let need = meta_len - meta_bytes.len();
                    if let Ok(n) = stream.read(&mut extra[..need.min(256)]) {
                        meta_bytes.extend_from_slice(&extra[..n]);
                    } else {
                        break;
                    }
                }

                if let Ok(meta_str) = String::from_utf8(meta_bytes.iter().take_while(|&&b| b != 0).cloned().collect()) {
                    if let Some(title) = parse_stream_title(&meta_str) {
                        let mut np = now_playing.lock().unwrap();
                        *np = title;
                    }
                }
            }

            *bytes_until_meta = metaint;
        } else {
            let audio_bytes = if metaint > 0 {
                (data.len() - i).min(*bytes_until_meta)
            } else {
                data.len() - i
            };

            {
                let mut buf = buffer.lock().unwrap();
                if buf.len() < 1_000_000 {
                    buf.extend(&data[i..i + audio_bytes]);
                }
            }

            if metaint > 0 {
                *bytes_until_meta -= audio_bytes;
            }
            i += audio_bytes;
        }
    }

    Ok(())
}

fn parse_stream_title(meta: &str) -> Option<String> {
    if let Some(start) = meta.find("StreamTitle='") {
        let rest = &meta[start + 13..];
        if let Some(end) = rest.find("';") {
            let title = &rest[..end];
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
    }
    None
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    for i in 0..data.len().saturating_sub(3) {
        if &data[i..i+4] == b"\r\n\r\n" {
            return Some(i);
        }
    }
    None
}

// FFT Visualizer
struct Visualizer {
    fft_buffer: Arc<Mutex<Vec<f32>>>,
    cache: Cache,
}

impl Visualizer {
    fn new(fft_buffer: Arc<Mutex<Vec<f32>>>) -> Self {
        Self {
            fft_buffer,
            cache: Cache::new(),
        }
    }

    fn request_redraw(&mut self) {
        self.cache.clear();
    }
}

impl<Message> Program<Message> for Visualizer {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            let magnitudes = self.fft_buffer.lock().unwrap().clone();
            if magnitudes.is_empty() {
                return;
            }

            let bar_width = bounds.width / NUM_BARS as f32;
            let gap = 2.0;
            let max_height = bounds.height;

            for (i, &mag) in magnitudes.iter().enumerate() {
                let height = (mag.min(1.0) * max_height).max(2.0);
                let x = i as f32 * bar_width + gap / 2.0;
                let y = bounds.height - height;

                // Gradient color based on height (cyan to magenta)
                let t = height / max_height;
                let color = Color::from_rgb(
                    0.2 + t * 0.7,      // R: cyan to magenta
                    0.8 - t * 0.5,      // G: high to low
                    1.0,                 // B: always high
                );

                let bar = Path::rectangle(
                    iced::Point::new(x, y),
                    iced::Size::new(bar_width - gap, height),
                );
                frame.fill(&bar, color);

                // Add glow effect at top
                if height > 5.0 {
                    let glow = Path::rectangle(
                        iced::Point::new(x, y),
                        iced::Size::new(bar_width - gap, 3.0),
                    );
                    frame.fill(&glow, Color::from_rgba(1.0, 1.0, 1.0, 0.5));
                }
            }
        });

        vec![geometry]
    }
}

struct PartyFMPlayer {
    state: StreamState,
    audio: Option<AudioController>,
    status_rx: Arc<Mutex<Option<Receiver<StreamState>>>>,
    logo_handle: image::Handle,
    smiley_handle: image::Handle,
    auto_started: bool,
    now_playing: Arc<Mutex<String>>,
    fft_buffer: Arc<Mutex<Vec<f32>>>,
    visualizer: Visualizer,
}

impl Default for PartyFMPlayer {
    fn default() -> Self {
        let fft_buffer = Arc::new(Mutex::new(vec![0.0; NUM_BARS]));
        Self {
            state: StreamState::Stopped,
            audio: None,
            status_rx: Arc::new(Mutex::new(None)),
            logo_handle: image::Handle::from_bytes(LOGO_BYTES.to_vec()),
            smiley_handle: image::Handle::from_bytes(SMILEY_BYTES.to_vec()),
            auto_started: false,
            now_playing: Arc::new(Mutex::new(String::new())),
            fft_buffer: fft_buffer.clone(),
            visualizer: Visualizer::new(fft_buffer),
        }
    }
}

impl PartyFMPlayer {
    fn subscription(&self) -> Subscription<Message> {
        Subscription::run(|| {
            iced::stream::channel(10, |mut output| async move {
                loop {
                    tokio::time::sleep(Duration::from_millis(33)).await; // ~30 FPS
                    let _ = output.try_send(Message::Tick);
                }
            })
        })
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        if !self.auto_started {
            self.auto_started = true;
            if self.audio.is_none() {
                let (tx, rx) = mpsc::channel();
                *self.status_rx.lock().unwrap() = Some(rx);
                self.audio = Some(AudioController::new(tx, self.now_playing.clone(), self.fft_buffer.clone()));
            }
            if let Some(ref controller) = self.audio {
                controller.play();
            }
            self.state = StreamState::Connecting;
        }

        if let Ok(rx_guard) = self.status_rx.lock() {
            if let Some(ref rx) = *rx_guard {
                while let Ok(status) = rx.try_recv() {
                    self.state = status;
                }
            }
        }

        match message {
            Message::TogglePlay => {
                match self.state {
                    StreamState::Stopped | StreamState::Error(_) => {
                        if self.audio.is_none() {
                            let (tx, rx) = mpsc::channel();
                            *self.status_rx.lock().unwrap() = Some(rx);
                            self.audio = Some(AudioController::new(tx, self.now_playing.clone(), self.fft_buffer.clone()));
                        }
                        if let Some(ref controller) = self.audio {
                            controller.play();
                        }
                        self.state = StreamState::Connecting;
                    }
                    StreamState::Playing | StreamState::Connecting => {
                        if let Some(ref controller) = self.audio {
                            controller.stop();
                        }
                        self.state = StreamState::Stopped;
                    }
                }
            }
            Message::VolumeUp => {
                if let Some(ref controller) = self.audio {
                    controller.volume_up();
                }
            }
            Message::VolumeDown => {
                if let Some(ref controller) = self.audio {
                    controller.volume_down();
                }
            }
            Message::Tick => {
                self.visualizer.request_redraw();
            }
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let gold = Color::from_rgb(0.95, 0.75, 0.1);
        let white = Color::from_rgb(1.0, 1.0, 1.0);
        let light_blue = Color::from_rgb(0.6, 0.85, 1.0);

        let logo = image(self.logo_handle.clone())
            .width(Length::Fixed(380.0))
            .height(Length::Fixed(62.0));

        let smiley = image(self.smiley_handle.clone())
            .width(Length::Fixed(35.0))
            .height(Length::Fixed(35.0));

        let now_playing_text = {
            let np = self.now_playing.lock().unwrap();
            if np.is_empty() { "".to_string() } else { np.clone() }
        };

        let (status_text, status_color) = match &self.state {
            StreamState::Stopped => ("Klar til at lytte!", Color::from_rgb(0.6, 0.6, 0.7)),
            StreamState::Connecting => ("Forbinder...", gold),
            StreamState::Playing => ("ON AIR", Color::from_rgb(0.2, 1.0, 0.4)),
            StreamState::Error(e) => (e.as_str(), Color::from_rgb(1.0, 0.3, 0.3)),
        };

        let status_label = text(status_text).size(16).color(status_color);

        let status_row = row![smiley, Space::with_width(10.0), status_label]
            .align_y(alignment::Vertical::Center);

        let now_playing_display = if !now_playing_text.is_empty() && self.state == StreamState::Playing {
            let display_text = if now_playing_text.len() > 45 {
                format!("{}...", &now_playing_text[..42])
            } else {
                now_playing_text
            };
            text(display_text).size(13).color(gold)
        } else {
            text("").size(13)
        };

        // FFT Visualizer canvas
        let visualizer_canvas = canvas(&self.visualizer)
            .width(Length::Fixed(380.0))
            .height(Length::Fixed(70.0));

        let play_btn_text = if self.state == StreamState::Playing || self.state == StreamState::Connecting {
            "■  STOP"
        } else {
            "▶  LYT MED!"
        };

        let is_playing = self.state == StreamState::Playing;

        let play_button = button(
            container(text(play_btn_text).size(20).color(if is_playing { white } else { Color::from_rgb(0.1, 0.1, 0.1) }))
                .center_x(Length::Fill)
                .center_y(Length::Fill)
        )
        .width(Length::Fixed(180.0))
        .height(Length::Fixed(50.0))
        .style(move |_theme, status| {
            let base = if is_playing {
                Color::from_rgb(0.7, 0.2, 0.2)
            } else {
                Color::from_rgb(0.95, 0.75, 0.1)
            };

            let bg = match status {
                button::Status::Hovered => Color::from_rgb(
                    (base.r * 1.15).min(1.0),
                    (base.g * 1.15).min(1.0),
                    (base.b * 1.15).min(1.0),
                ),
                button::Status::Pressed => Color::from_rgb(base.r * 0.8, base.g * 0.8, base.b * 0.8),
                _ => base,
            };

            button::Style {
                background: Some(iced::Background::Gradient(iced::Gradient::Linear(
                    iced::gradient::Linear::new(std::f32::consts::PI / 2.0)
                        .add_stop(0.0, Color::from_rgb(bg.r * 1.2, bg.g * 1.2, bg.b * 1.2))
                        .add_stop(0.5, bg)
                        .add_stop(1.0, Color::from_rgb(bg.r * 0.7, bg.g * 0.7, bg.b * 0.7))
                ))),
                text_color: if is_playing { white } else { Color::from_rgb(0.1, 0.1, 0.1) },
                border: Border { color: Color::from_rgb(0.3, 0.3, 0.3), width: 2.0, radius: 25.0.into() },
                shadow: Shadow { color: Color::from_rgba(0.0, 0.0, 0.0, 0.5), offset: iced::Vector::new(0.0, 4.0), blur_radius: 8.0 },
            }
        })
        .on_press(Message::TogglePlay);

        let vol_btn_style = move |_theme: &iced::Theme, status: button::Status| {
            let base = Color::from_rgb(0.15, 0.15, 0.2);
            let bg = match status {
                button::Status::Hovered => Color::from_rgb(0.25, 0.25, 0.35),
                button::Status::Pressed => Color::from_rgb(0.1, 0.1, 0.15),
                _ => base,
            };
            button::Style {
                background: Some(iced::Background::Gradient(iced::Gradient::Linear(
                    iced::gradient::Linear::new(std::f32::consts::PI / 2.0)
                        .add_stop(0.0, Color::from_rgb(0.3, 0.3, 0.35))
                        .add_stop(0.5, bg)
                        .add_stop(1.0, Color::from_rgb(0.08, 0.08, 0.1))
                ))),
                text_color: light_blue,
                border: Border { color: Color::from_rgb(0.4, 0.4, 0.5), width: 2.0, radius: 18.0.into() },
                shadow: Shadow::default(),
            }
        };

        let vol_down = button(
            container(text("-").size(26).color(light_blue)).center_x(Length::Fill).center_y(Length::Fill)
        )
        .width(Length::Fixed(45.0))
        .height(Length::Fixed(45.0))
        .style(vol_btn_style)
        .on_press(Message::VolumeDown);

        let vol_up = button(
            container(text("+").size(22).color(light_blue)).center_x(Length::Fill).center_y(Length::Fill)
        )
        .width(Length::Fixed(45.0))
        .height(Length::Fixed(45.0))
        .style(vol_btn_style)
        .on_press(Message::VolumeUp);

        let volume_row = row![vol_down, Space::with_width(12.0), play_button, Space::with_width(12.0), vol_up]
            .align_y(alignment::Vertical::Center);

        let stream_info = text("128 kbps • stream1.partyfm.dk").size(10).color(Color::from_rgb(0.4, 0.5, 0.6));

        let content = column![
            vertical_space().height(6.0),
            logo,
            vertical_space().height(10.0),
            status_row,
            vertical_space().height(4.0),
            now_playing_display,
            vertical_space().height(10.0),
            visualizer_canvas,
            vertical_space().height(12.0),
            volume_row,
            vertical_space().height(10.0),
            stream_info,
            vertical_space().height(6.0),
        ]
        .align_x(alignment::Horizontal::Center)
        .width(Length::Fill);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(move |_theme| {
                container::Style {
                    background: Some(iced::Background::Gradient(iced::Gradient::Linear(
                        iced::gradient::Linear::new(std::f32::consts::PI)
                            .add_stop(0.0, Color::from_rgb(0.05, 0.08, 0.15))
                            .add_stop(0.3, Color::from_rgb(0.02, 0.02, 0.05))
                            .add_stop(1.0, Color::from_rgb(0.0, 0.0, 0.0))
                    ))),
                    border: Border { color: Color::from_rgb(0.2, 0.3, 0.4), width: 1.0, radius: 0.0.into() },
                    text_color: Some(white),
                    shadow: Shadow::default(),
                }
            })
            .into()
    }

    fn theme(&self) -> iced::Theme {
        iced::Theme::custom(
            "PartyFM".to_string(),
            Palette {
                background: Color::from_rgb(0.0, 0.0, 0.0),
                text: Color::from_rgb(1.0, 1.0, 1.0),
                primary: Color::from_rgb(0.95, 0.75, 0.1),
                success: Color::from_rgb(0.2, 1.0, 0.4),
                danger: Color::from_rgb(0.9, 0.2, 0.2),
            },
        )
    }
}
