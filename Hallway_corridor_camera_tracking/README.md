# Hallway_Corridor_Camera_Tracking
Задача: Необходимо реализовать трекер людей с отображением пройденного пути на кадре длиной N пикселей. 
За опорную точку взять нижнюю среднюю точку bounding box.

Дополнительно: сделать отображение пути сверху, bird-eye view. Использовать для этого матрицу гомографии.

Usage example:
```bash
python main.py --video_name video/004.avi --n_frames 10 --get_coord no
```
В этом [датасете](http://www.santhoshsunderrajan.com/datasets.html#hfh_tracking) были даны матрицы гомографии для видео,но они оказались не правильными, поэтому я получаю их с помощью cv2.findHomography. Для этого, выбираются ключевые точки пола на изображении, и с помощью функции get_homography_matrix() получается  необходимая матрица.

*Пример выбранных точек для 4-ого видео* 

![Screenshot](additional/Selected_points.png)

Результат тестового задания продемонтрирован на видео, левая часть видео -- это трэкер людей с отображением пройденного пути на кадре длиной N пикселей,
а правая -- это отображение пройденного пути сверху(bird-eye view).

https://github.com/MaximKondakov/Hallway_Corridor_Camera_Tracking/assets/85742231/352659cb-f444-40ba-b0b7-3b1020777e5b

