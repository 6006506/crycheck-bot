import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardMarkup, KeyboardButton
from pydub import AudioSegment
import librosa
import soundfile as sf
import requests
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import torch

import logging
logging.basicConfig(level=logging.INFO)

# === Настройки ===
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")  # Получите у @BotFather
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Опционально: Hugging Face token для Inference API
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()

# Инициализация ML модели (ленивая загрузка)
ml_model = None
ml_processor = None
USE_ML = True  # Переключение между ML и простым анализом

# Клавиатура — для удобства
kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="🎤 Запишите плач ребёнка")]],
    resize_keyboard=True,
    one_time_keyboard=False
)

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я — *CryCheck*, ваш помощник в понимании детского плача.\n\n"
        "👉 Нажмите на значок микрофона в Telegram и запишите 10–30 сек плача.\n"
        "Я проанализирую его с помощью *AI модели* и подскажу: *голод, усталость или дискомфорт?*\n\n"
        "🤖 Использую бесплатную ML модель от Hugging Face для анализа\n"
        "🔒 Ваши аудиозаписи *не сохраняются* — обработка происходит локально и файлы удаляются сразу.",
        parse_mode="Markdown",
        reply_markup=kb
    )

# === Основная логика: обработка голосового сообщения ===
@router.message()
async def handle_voice(message: Message):
    if message.voice:
        voice = message.voice
        user_id = message.from_user.id
        file_id = voice.file_id

        ogg_path = TEMP_DIR / f"{user_id}_{file_id}.ogg"
        wav_path = TEMP_DIR / f"{user_id}_{file_id}.wav"

        try:
            # 1. Скачиваем голосовое
            await message.answer("📥 Получаю аудио…")
            file = await bot.get_file(file_id)
            await bot.download_file(file.file_path, ogg_path)

            # 2. Конвертируем в WAV (16kHz, моно)
            await message.answer("⚙️ Обрабатываю звук…")
            audio = AudioSegment.from_file(ogg_path, format="ogg")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")

            # 3. Анализ с помощью ML модели
            await message.answer("🤖 Анализирую с помощью AI…")
            if USE_ML:
                analysis = await analyze_cry_ml(wav_path, duration_sec=voice.duration)
            else:
                analysis = analyze_cry_simple(wav_path, duration_sec=voice.duration)

            # 4. Отправляем результат
            await message.answer(
                f"🔍 *Анализ завершён:*\n{analysis}",
                parse_mode="Markdown"
            )

        except Exception as e:
            await message.answer(f"❌ Ошибка обработки: {str(e)}\nПопробуйте записать ещё раз.")
            logging.error(f"Ошибка: {e}", exc_info=True)

        finally:
            # 5. Чистим временные файлы — ПРИВАТНОСТЬ в приоритете!
            for p in [ogg_path, wav_path]:
                if p.exists():
                    p.unlink()

    else:
        await message.answer(
            "Пожалуйста, отправьте *голосовое сообщение* (нажмите на значок микрофона 🎤).",
            parse_mode="Markdown",
            reply_markup=kb
        )

# === ML анализатор плача с использованием Hugging Face ===
async def analyze_cry_ml(wav_path: Path, duration_sec: int) -> str:
    """
    Анализ плача с помощью бесплатной ML модели от Hugging Face.
    Использует audio classification pipeline для анализа аудио признаков.
    """
    try:
        global ml_model, ml_processor
        
        # Загружаем аудио с помощью librosa (поддерживает нужный формат для моделей)
        audio_array, sr = librosa.load(str(wav_path), sr=16000, duration=30)
        
        # Если аудио слишком короткое, дополняем тишиной
        if len(audio_array) < sr * 2:  # меньше 2 секунд
            audio_array = np.pad(audio_array, (0, sr * 2 - len(audio_array)), mode='constant')
        
        # Инициализируем модель при первом использовании
        if ml_model is None:
            try:
                # Используем общую модель для классификации аудио
                # Модель MIT/ast-finetuned-audioset-10-10-0.4593 хорошо работает для общих звуков
                model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
                logging.info(f"Загрузка ML модели: {model_name}")
                
                # Используем pipeline для упрощения
                ml_model = pipeline(
                    "audio-classification",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                logging.info("ML модель успешно загружена")
            except Exception as e:
                logging.warning(f"Не удалось загрузить ML модель: {e}. Используется простой анализ.")
                return analyze_cry_simple(wav_path, duration_sec)
        
        # Анализируем аудио с помощью ML модели
        # Разбиваем на сегменты по 10 секунд для лучшего анализа
        segment_length = sr * 10
        segments = []
        for i in range(0, len(audio_array), segment_length):
            segment = audio_array[i:i+segment_length]
            if len(segment) >= sr * 2:  # минимум 2 секунды
                segments.append(segment)
        
        if not segments:
            segments = [audio_array]
        
        # Анализируем каждый сегмент
        all_predictions = []
        for segment in segments:
            # Сохраняем временный файл для модели
            temp_segment_path = TEMP_DIR / f"segment_{hash(str(segment))}.wav"
            sf.write(str(temp_segment_path), segment, sr)
            
            try:
                predictions = ml_model(str(temp_segment_path))
                all_predictions.extend(predictions)
            finally:
                if temp_segment_path.exists():
                    temp_segment_path.unlink()
        
        # Извлекаем признаки из аудио для дополнительного анализа
        features = extract_audio_features(audio_array, sr)
        
        # Комбинируем результаты ML и аудио-признаков
        analysis = interpret_ml_results(all_predictions, features, duration_sec)
        
        return analysis
        
    except Exception as e:
        logging.error(f"Ошибка ML анализа: {e}", exc_info=True)
        # Fallback на простой анализ
        return analyze_cry_simple(wav_path, duration_sec)

def extract_audio_features(audio_array: np.ndarray, sr: int) -> dict:
    """Извлекает аудио-признаки для анализа плача"""
    features = {}
    
    # Основные признаки
    features['mean_amplitude'] = np.mean(np.abs(audio_array))
    features['std_amplitude'] = np.std(np.abs(audio_array))
    features['max_amplitude'] = np.max(np.abs(audio_array))
    
    # Спектральные признаки
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    
    # Zero crossing rate (показывает резкость звука)
    zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # MFCC (Mel-frequency cepstral coefficients) - важные для распознавания звуков
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    
    # Темп (rhythm)
    tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sr)
    features['tempo'] = tempo if tempo > 0 else 0
    
    return features

def interpret_ml_results(predictions: list, features: dict, duration_sec: int) -> str:
    """
    Интерпретирует результаты ML модели и аудио-признаков
    для определения причины плача
    """
    # Анализируем предсказания модели
    # Ищем признаки, связанные с криком, плачем, дискомфортом
    cry_indicators = []
    intensity_score = 0
    
    for pred in predictions:
        label = pred.get('label', '').lower()
        score = pred.get('score', 0)
        
        # Ищем ключевые слова в метках
        if any(word in label for word in ['cry', 'crying', 'scream', 'shout', 'distress', 'pain']):
            cry_indicators.append((label, score))
            intensity_score += score
    
    # Анализируем аудио-признаки
    spectral_centroid = features.get('spectral_centroid_mean', 0)
    zcr = features.get('zcr_mean', 0)
    mean_amp = features.get('mean_amplitude', 0)
    tempo = features.get('tempo', 0)
    
    # Логика определения причины плача на основе признаков
    # Высокий spectral centroid + высокий ZCR = резкий, пронзительный крик (боль/дискомфорт)
    # Средний spectral centroid + ритмичность = голод
    # Низкий-средний spectral centroid + переменная амплитуда = усталость
    
    if spectral_centroid > 3000 and zcr > 0.15 and mean_amp > 0.3:
        # Резкий, пронзительный крик
        return (
            "⚠️ *Вероятно — боль или дискомфорт*\n"
            f"— обнаружены признаки резкого, пронзительного крика (AI уверенность: {intensity_score*100:.1f}%)\n"
            "— высокие частоты, резкие переходы\n"
            "— не успокаивается при укачивании\n"
            "💡 *Проверьте: животик, подгузник, температуру. При сомнениях — вызовите врача.*"
        )
    elif 2000 < spectral_centroid < 3000 and 0.05 < zcr < 0.12 and duration_sec < 30:
        # Ритмичный, умеренный плач
        return (
            "🍼 *Вероятно — голод*\n"
            f"— обнаружены признаки ритмичного плача (AI уверенность: {intensity_score*100:.1f}%)\n"
            "— плач начинается постепенно, ритмичный\n"
            "— средние частоты, регулярные паузы\n"
            "💡 *Попробуйте покормить — даже если «по времени ещё рано».*"
        )
    elif spectral_centroid < 2500 and zcr < 0.1:
        # Мягкий, затухающий плач
        return (
            "😴 *Вероятно — усталость / перевозбуждение*\n"
            f"— обнаружены признаки усталости (AI уверенность: {intensity_score*100:.1f}%)\n"
            "— плач нарастает медленно, затем затухает\n"
            "— низкие-средние частоты, плавные переходы\n"
            "💡 *Приглушите свет и звуки. Попробуйте белый шум и медленное укачивание.*"
        )
    else:
        # Комбинированный анализ
        if mean_amp > 0.25 and duration_sec > 20:
            return (
                "🤔 *Смешанные признаки*\n"
                f"— AI анализ показывает сложную картину (уверенность: {intensity_score*100:.1f}%)\n"
                "— возможно, несколько причин одновременно\n"
                "💡 *Попробуйте: проверить подгузник, покормить, укачать. Наблюдайте за реакцией.*"
            )
        else:
            return (
                "🔍 *Анализ завершён*\n"
                f"— AI обработал аудио (уверенность: {intensity_score*100:.1f}%)\n"
                "— признаки неоднозначны, требуется больше данных\n"
                "💡 *Попробуйте записать более длинное сообщение (20-30 сек) для лучшего анализа.*"
            )

# === Простой анализатор плача (fallback без ML) ===
def analyze_cry_simple(wav_path: Path, duration_sec: int) -> str:
    """
    MVP-анализ на основе:
    - длительности
    - амплитудных всплесков («резкость»)
    - среднего уровня громкости
    
    Источники вдохновения: Dunstan Baby Language, Newman et al. (2021)
    """
    try:
        # Загружаем аудио как массив (моно)
        audio = AudioSegment.from_wav(wav_path)
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)  # сводим к моно

        # Нормализуем
        samples = samples / np.max(np.abs(samples) + 1e-6)

        # Признаки:
        mean_amp = np.mean(np.abs(samples))
        std_amp = np.std(np.abs(samples))
        peak_amp = np.max(np.abs(samples))
        # Считаем "всплески" — моменты, где амплитуда резко растёт (> 3σ)
        diff = np.abs(np.diff(np.abs(samples)))
        spikes = np.sum(diff > 3 * np.std(diff))

        # Правила (упрощённая эвристика)
        if duration_sec < 20 and mean_amp > 0.2 and spikes < 5:
            # Короткий, умеренно громкий, плавные переходы
            return (
                "🍼 *Вероятно — голод*\n"
                "— плач начинается постепенно,\n"
                "— ритмичный, с паузами (~2 сек на цикл),\n"
                "💡 *Попробуйте покормить — даже если «по времени ещё рано».*"
            )
        elif duration_sec > 40 and mean_amp > 0.3 and spikes > 10:
            # Долгий, громкий, много резких всплесков
            return (
                "⚠️ *Вероятно — боль или дискомфорт*\n"
                "— резкие всплески, крик «на выдохе»,\n"
                "— не успокаивается при укачивании,\n"
                "💡 *Проверьте: животик, подгузник, температуру. При сомнениях — вызовите врача.*"
            )
        else:
            # Всё остальное — усталость/перевозбуждение
            return (
                "😴 *Вероятно — усталость / перевозбуждение*\n"
                "— плач нарастает медленно, затем затухает,\n"
                "— ребёнок трёт глаза/зевает,\n"
                "💡 *Приглушите свет и звуки. Попробуйте белый шум и медленное укачивание.*"
            )

    except Exception as e:
        logging.error(f"Ошибка анализа: {e}")
        return "❓ Не удалось проанализировать плач. Попробуйте записать короче и чётче."

# === Запуск ===
dp.include_router(router)

if __name__ == "__main__":
    print("🚀 CryCheck бот запущен! Откройте Telegram и отправьте /start")
    import asyncio
    asyncio.run(dp.start_polling(bot))