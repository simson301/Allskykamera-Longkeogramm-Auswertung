#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse eines LongTerm-Keogramms.

v4:
- Keogramm oben korrekt dargestellt
- Summe und Mittelwert getrennt darunter
- Sonnenaufgang grün, Sonnenuntergang rot
- Soll-/Ist-Linienanzahl berücksichtigt den aktuellen Monat:
  * Vergangene Monate: Soll = kompletter Monat
  * Aktueller Monat: Soll = nur vom Monatsanfang bis jetzt, gerundet auf das Intervall
  * Zukünftige Monate: Soll = 0

Benötigte Pakete:
    py -m pip install pillow numpy matplotlib astral

Beispiel:
    py analyse_longkeogram_mit_keogramm_und_sonne_v4.py LongKeogram_202606.jpg ^
        --start "2026-06-01 00:00:00" ^
        --interval 10 ^
        --lat 52.52 ^
        --lon 13.405 ^
        --timezone Europe/Berlin

Optional:
    --expected-mode auto        Default. Aktueller Monat nur bis jetzt, alte Monate vollständig.
    --expected-mode full-month  Immer kompletter Monat.
    --expected-mode until-end   Soll bis zur letzten Bildlinie; praktisch keine Vollständigkeitsprüfung.
"""

import argparse
import calendar
import csv
from datetime import datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from astral import LocationInfo
from astral.sun import sun


def parse_start_time(value):
    print(value)
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    raise ValueError(
        "Startzeit konnte nicht gelesen werden. Beispiel: 2026-06-01 00:00:00"
    )


def get_now_local_naive(timezone_name):
    """
    Liefert die aktuelle lokale Zeit als naive datetime.
    Für Python >= 3.9 wird zoneinfo genutzt.
    Falls zoneinfo nicht verfügbar ist, wird datetime.now() verwendet.
    """
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(timezone_name)).replace(tzinfo=None)
        except Exception:
            pass

    return datetime.now()


def floor_time_to_interval(start_time, now_time, interval_minutes):
    """
    Rundet now_time nach unten auf das nächste gültige Intervall
    bezogen auf start_time.
    """
    if now_time <= start_time:
        return start_time

    delta_minutes = int((now_time - start_time).total_seconds() // 60)
    steps = delta_minutes // interval_minutes
    return start_time + timedelta(minutes=steps * interval_minutes)


def get_sun_events_for_range(start_dt, end_dt, latitude, longitude, timezone_name):
    if ZoneInfo is None:
        raise RuntimeError(
            "zoneinfo ist nicht verfügbar. Nutze Python >= 3.9 oder installiere backports.zoneinfo."
        )

    tz = ZoneInfo(timezone_name)
    location = LocationInfo(
        name="Standort",
        region="",
        timezone=timezone_name,
        latitude=latitude,
        longitude=longitude,
    )

    day_count = (end_dt.date() - start_dt.date()).days + 1
    sunrise_list = []
    sunset_list = []

    for i in range(day_count):
        current_date = start_dt.date() + timedelta(days=i)
        s = sun(location.observer, date=current_date, tzinfo=tz)

        sunrise = s.get("sunrise")
        sunset = s.get("sunset")

        if sunrise is not None:
            sunrise_naive = sunrise.replace(tzinfo=None)
            if start_dt <= sunrise_naive <= end_dt:
                sunrise_list.append(sunrise_naive)

        if sunset is not None:
            sunset_naive = sunset.replace(tzinfo=None)
            if start_dt <= sunset_naive <= end_dt:
                sunset_list.append(sunset_naive)

    return sunrise_list, sunset_list


def calc_month_line_stats(start_time, interval_minutes, actual_width, timezone_name, expected_mode):
    year = start_time.year
    month = start_time.month
    days_in_month = calendar.monthrange(year, month)[1]

    month_start = datetime(year, month, 1, 0, 0, 0)
    next_month_start = month_start + timedelta(days=days_in_month)
    full_month_last_line = next_month_start - timedelta(minutes=interval_minutes)

    actual_lines = int(actual_width)
    actual_end = start_time + timedelta(minutes=(actual_lines - 1) * interval_minutes)

    now_local = get_now_local_naive(timezone_name)
    current_month_start = datetime(now_local.year, now_local.month, 1, 0, 0, 0)

    if expected_mode == "full-month":
        expected_end = full_month_last_line
        expected_reason = "voller Monat"
    elif expected_mode == "until-end":
        expected_end = actual_end
        expected_reason = "bis zur letzten vorhandenen Bildlinie"
    else:
        # auto:
        # - vergangener Monat: voller Monat
        # - aktueller Monat: bis jetzt, auf das Intervall gerundet
        # - zukünftiger Monat: 0 Soll-Linien
        if month_start < current_month_start:
            expected_end = full_month_last_line
            expected_reason = "vergangener Monat, voller Monat"
        elif month_start == current_month_start:
            expected_end = floor_time_to_interval(
                start_time=start_time,
                now_time=now_local,
                interval_minutes=interval_minutes,
            )
            # Nicht über das Monatsende hinauslaufen
            if expected_end > full_month_last_line:
                expected_end = full_month_last_line
            expected_reason = "aktueller Monat, bis jetzt"
        else:
            expected_end = None
            expected_reason = "zukünftiger Monat"

    if expected_end is None or expected_end < start_time:
        expected_lines = 0
    else:
        expected_lines = int(((expected_end - start_time).total_seconds() // 60) // interval_minutes) + 1

    diff = actual_lines - expected_lines
    missing_lines = max(0, expected_lines - actual_lines)
    extra_lines = max(0, actual_lines - expected_lines)

    missing_minutes = missing_lines * interval_minutes
    missing_hours = missing_minutes // 60
    missing_rest_minutes = missing_minutes % 60

    extra_minutes = extra_lines * interval_minutes
    extra_hours = extra_minutes // 60
    extra_rest_minutes = extra_minutes % 60

    coverage_pct = (actual_lines / expected_lines * 100.0) if expected_lines else 0.0

    return {
        "days_in_month": days_in_month,
        "expected_lines": expected_lines,
        "actual_lines": actual_lines,
        "diff": diff,
        "missing_lines": missing_lines,
        "extra_lines": extra_lines,
        "missing_minutes": missing_minutes,
        "missing_hours": missing_hours,
        "missing_rest_minutes": missing_rest_minutes,
        "extra_minutes": extra_minutes,
        "extra_hours": extra_hours,
        "extra_rest_minutes": extra_rest_minutes,
        "coverage_pct": coverage_pct,
        "expected_end": expected_end,
        "actual_end": actual_end,
        "now_local": now_local,
        "expected_reason": expected_reason,
        "expected_mode": expected_mode,
    }


def analyse_longkeogram(
    image_path,
    start,
    end,
    absolute_start="2026-06-01 00:00:00",
    interval_minutes=10,
    latitude=None,
    longitude=None,
    timezone_name="Europe/Berlin",
    output_prefix=None,
    expected_mode="auto",
    sections = [],
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError("Datei nicht gefunden: {}".format(image_path))

    absolute_start_time = parse_start_time(absolute_start)
    start_time = parse_start_time(start)
    end_time = parse_start_time(end)

    start_row = int((start_time - absolute_start_time).total_seconds() // 60) // interval_minutes
    end_row = int((end_time - absolute_start_time).total_seconds() // 60) // interval_minutes
    

    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    img = img.crop((start_row, 0, end_row, height))
    width, height = img.size
    rgb_display = np.asarray(img, dtype=np.uint8)
    rgb_float = rgb_display.astype(np.float32)

    
    timestamps = [
        start_time + timedelta(minutes=i * interval_minutes)
        for i in range(width)
    ]

    #end_time = timestamps[-1]

    line_stats = calc_month_line_stats(
        start_time=start_time,
        interval_minutes=interval_minutes,
        actual_width=width,
        timezone_name=timezone_name,
        expected_mode=expected_mode,
    )

    sunrise_list = []
    sunset_list = []
    if latitude is not None and longitude is not None:
        sunrise_list, sunset_list = get_sun_events_for_range(
            start_dt=start_time,
            end_dt=end_time,
            latitude=latitude,
            longitude=longitude,
            timezone_name=timezone_name,
        )

    plot_long_keogram_analasys(
        y0=0,
        y1=height,
        img=img,
        overlay=False,
        rgb_float=rgb_float,
        img_path=image_path,
        output_prefix=output_prefix,
        timestamps=timestamps,
        interval_minutes=interval_minutes,
        line_stats=line_stats,
        latitude=latitude,
        longitude=longitude,
        timezone_name=timezone_name,
        sunrise_list=sunrise_list,
        sunset_list=sunset_list,
    )

    for section in sections:
        y0 = section[0]
        y1 = section[1]
        if y1 > height:
            y1 = height
        elif y1 <= 0:
            y1 = height
        elif y1 == -1:
            y1 = height
        if y0 > height:
            y0 = 0
        elif y0 < 0:
            y0 = 0
        if y0 > y1:
            y0, y1 = y1, y0
        
        plot_long_keogram_analasys(
            y0=y0,
            y1=y1,
            img=img,
            overlay=True,
            rgb_float=rgb_float,
            img_path=image_path,
            output_prefix=output_prefix,
            timestamps=timestamps,
            interval_minutes=interval_minutes,
            line_stats=line_stats,
            latitude=latitude,
            longitude=longitude,
            timezone_name=timezone_name,
            sunrise_list=sunrise_list,
            sunset_list=sunset_list,
        )

def plot_long_keogram_analasys(y0, y1, img, overlay, rgb_float, img_path, output_prefix, timestamps, interval_minutes, line_stats, latitude, longitude, timezone_name, sunrise_list, sunset_list):
    image_path = Path(img_path)

    print(y0, y1)

    if not image_path.exists():
        raise FileNotFoundError("Datei nicht gefunden: {}".format(image_path))

    if output_prefix is None:
        output_prefix = image_path.stem + "_helligkeit"

    output_csv = image_path.with_name(f"{output_prefix}_{y0}-{y1}.csv")
    output_plot = image_path.with_name(f"{output_prefix}_{y0}-{y1}.png")

    width, height = img.size

    luminance = (
        0.2126 * rgb_float[y0:y1, :, 0]
        + 0.7152 * rgb_float[y0:y1, :, 1]
        + 0.0722 * rgb_float[y0:y1, :, 2]
    )

    brightness_sum = luminance.sum(axis=0)
    brightness_mean = luminance.mean(axis=0)
    
    r_sum = (rgb_float[y0:y1, :, 0]).sum(axis=0)
    g_sum = (rgb_float[y0:y1, :, 1]).sum(axis=0)
    b_sum = (rgb_float[y0:y1, :, 2]).sum(axis=0)

    r_mean = (rgb_float[y0:y1, :, 0]).mean(axis=0)
    g_mean = (rgb_float[y0:y1, :, 1]).mean(axis=0)
    b_mean = (rgb_float[y0:y1, :, 2]).mean(axis=0)

    draw_img = img.convert("RGBA")
    # Transparente Overlay-Ebene erstellen
    if overlay:
        overlay = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(
            [(0, y0), (width, y1)],
            fill=(255, 0, 0, 80)  # Rot mit leichter Transparenz
        )

        draw_img = Image.alpha_composite(draw_img, overlay)


    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "brightness_sum",
            "brightness_mean",
            "red_sum",
            "green_sum",
            "blue_sum",
            "red_mean",
            "green_mean",
            "blue_mean",
        ])
        for ts, bsum, bmean, r_s, g_s, b_s, r_m, g_m, b_m in zip(timestamps, brightness_sum, brightness_mean, r_sum, g_sum, b_sum, r_mean, g_mean, b_mean):
            writer.writerow([
                ts.isoformat(sep=" "),
                float(bsum),
                float(bmean),
                float(r_s),
                float(g_s),
                float(b_s),
                float(r_m),  
                float(g_m),
                float(b_m),
            ])

    x_num = mdates.date2num(timestamps)
    x_min = x_num[0]
    x_max = x_num[-1]

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(
        5,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 2, 2]},
    )

    ax0.imshow(
        draw_img,
        aspect="auto",
        extent=[x_min, x_max, height, 0],
        interpolation="nearest",
    )
    ax0.set_ylabel("Pixelhöhe")
    ax0.set_title("LongKeogram")

    info_lines = [
        "Soll-Modus: {}".format(line_stats["expected_reason"]),
        "Jetzt: {}".format(line_stats["now_local"].strftime("%Y-%m-%d %H:%M")),
        "Monatstage: {}".format(line_stats["days_in_month"]),
        "Soll-Linien ({} min): {}".format(interval_minutes, line_stats["expected_lines"]),
        "Ist-Linien: {}".format(line_stats["actual_lines"]),
        "Abdeckung: {:.2f} %".format(line_stats["coverage_pct"]),
    ]

    if line_stats["expected_end"] is not None:
        info_lines.append(
            "Soll-Ende: {}".format(line_stats["expected_end"].strftime("%Y-%m-%d %H:%M"))
        )
    info_lines.append(
        "Ist-Ende:  {}".format(line_stats["actual_end"].strftime("%Y-%m-%d %H:%M"))
    )

    if line_stats["missing_lines"] > 0:
        info_lines.append(
            "Fehlend: {} Linien = {} h {} min".format(
                line_stats["missing_lines"],
                line_stats["missing_hours"],
                line_stats["missing_rest_minutes"],
            )
        )
    elif line_stats["extra_lines"] > 0:
        info_lines.append(
            "Mehr als Soll: {} Linien = {} h {} min".format(
                line_stats["extra_lines"],
                line_stats["extra_hours"],
                line_stats["extra_rest_minutes"],
            )
        )
    else:
        info_lines.append("Soll/Ist passt exakt")

    info_text = "\n".join(info_lines)

    ax0.text(
        0.01,
        0.98,
        info_text,
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    ax1.plot(timestamps, brightness_sum, label="Helligkeit Summe")
    ax1.set_ylabel("Summe")
    ax1.set_title("Helligkeitssumme pro vertikaler Linie")
    ax1.grid(True, alpha=0.3)

    ax2.plot(timestamps, brightness_mean, label="Helligkeit Mittelwert")
    ax2.set_ylabel("Mittelwert")
    ax2.set_xlabel("Zeit")
    ax2.set_title("Helligkeitsmittelwert pro vertikaler Linie")
    ax2.grid(True, alpha=0.3)

    ax3.plot(timestamps, r_sum, label="Rot", color='r')
    ax3.plot(timestamps, g_sum, label="Grün", color='g')
    ax3.plot(timestamps, b_sum, label="Blau", color='b')
    ax3.set_ylabel("Summe")
    ax3.set_xlabel("Zeit")
    ax3.set_title("Bereich: Hauswand - Farbsumme pro vertikaler Linie")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    ax4.plot(timestamps, r_mean, label="Rot", color='r')
    ax4.plot(timestamps, g_mean, label="Grün", color='g')
    ax4.plot(timestamps, b_mean, label="Blau", color='b')
    ax4.set_ylabel("Mittelwert")
    ax4.set_xlabel("Zeit")
    ax4.set_title("Bereich: Hauswand - Farbmittelwert pro vertikaler Linie")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    for sunrise in sunrise_list:
        for ax in (ax0, ax1, ax2, ax3, ax4):
            ax.axvline(sunrise, color="green", linestyle="--", linewidth=1, alpha=0.9)

    for sunset in sunset_list:
        for ax in (ax0, ax1, ax2, ax3, ax4):
            ax.axvline(sunset, color="red", linestyle="--", linewidth=1, alpha=0.9)

    if sunrise_list or sunset_list:
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="green", linestyle="--", label="Sonnenaufgang"),
            Line2D([0], [0], color="red", linestyle="--", label="Sonnenuntergang"),
        ]
        ax1.legend(handles=legend_elements, loc="upper right")

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    for ax in (ax0, ax1, ax2):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    plt.suptitle("LongKeogram-Auswertung", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=160)
    plt.close(fig)

    print("Bildgröße: {} x {} px".format(width, height))
    print("Startzeit: {}".format(timestamps[0]))
    print("Endzeit:   {}".format(timestamps[-1]))
    print("CSV:       {}".format(output_csv))
    print("Diagramm:  {}".format(output_plot))
    print("Soll-Modus: {}".format(line_stats["expected_reason"]))
    print("Jetzt: {}".format(line_stats["now_local"]))
    print("Soll-Linien ({} min): {}".format(interval_minutes, line_stats["expected_lines"]))
    print("Ist-Linien: {}".format(line_stats["actual_lines"]))
    print("Abdeckung: {:.2f} %".format(line_stats["coverage_pct"]))

    if line_stats["expected_end"] is not None:
        print("Soll-Ende: {}".format(line_stats["expected_end"]))
    print("Ist-Ende:  {}".format(line_stats["actual_end"]))

    if line_stats["missing_lines"] > 0:
        print(
            "Fehlende Linien: {} (= {} h {} min)".format(
                line_stats["missing_lines"],
                line_stats["missing_hours"],
                line_stats["missing_rest_minutes"],
            )
        )
    elif line_stats["extra_lines"] > 0:
        print(
            "Mehr als Soll: {} Linien (= {} h {} min)".format(
                line_stats["extra_lines"],
                line_stats["extra_hours"],
                line_stats["extra_rest_minutes"],
            )
        )
    else:
        print("Soll/Ist passt exakt.")

    if latitude is not None and longitude is not None:
        print("Standort:  lat={}, lon={}, timezone={}".format(latitude, longitude, timezone_name))
        print("Sonnenaufgänge im Bereich:   {}".format(len(sunrise_list)))
        print("Sonnenuntergänge im Bereich: {}".format(len(sunset_list)))
    else:
        print("Hinweis: Keine Sonnenauf-/untergänge berechnet, da --lat und --lon fehlen.")


def main():
    parser = argparse.ArgumentParser(
        description="Berechnet Summe und Mittelwert der Helligkeit pro vertikaler Linie eines LongKeogramms und zeigt das Keogramm darüber an."
    )
    parser.add_argument("image", help="Pfad zum LongKeogramm, z. B. LongKeogram_202606.jpg")
    parser.add_argument("--absolute-start", default="2026-06-01 00:00:00", help="Startzeit des Keogramms, Default: 2026-05-01 00:00:00")
    parser.add_argument("--interval", type=int, default=10, help="Intervall pro vertikaler Linie in Minuten, Default: 10")
    parser.add_argument("--lat", type=float, default=None, help="Breitengrad des Kamerastandorts")
    parser.add_argument("--lon", type=float, default=None, help="Längengrad des Kamerastandorts")
    parser.add_argument("--timezone", default="Europe/Berlin", help="Zeitzone, z. B. Europe/Berlin")
    parser.add_argument(
        "--expected-mode",
        default="auto",
        choices=["auto", "full-month", "until-end"],
        help="auto: aktueller Monat nur bis jetzt, alte Monate komplett; full-month: kompletter Monat; until-end: bis zur letzten Bildlinie",
    )
    parser.add_argument("--output-prefix", default=None, help="Optionaler Prefix für CSV und PNG")
    parser.add_argument(
    "--section",
    nargs=2,
    type=int,
    action="append",
    metavar=("X", "Y"),
    default=[],
    help="Analyse der Daten nur in der angegebenen Y-section",
    )
    parser.add_argument("--start", default="2026-05-21 00:00:00", help="Startzeitpunkt des untersuchten Intervalls, Default: 2026-05-01 00:00:00")
    parser.add_argument("--end", default="2026-05-29 00:00:00", help="Endzeitpunkt des untersuchten Intervalls, Default: 2026-05-29 00:00:00")
    args = parser.parse_args()

    analyse_longkeogram(
        image_path=args.image,
        start=args.start,
        end=args.end,
        absolute_start=args.absolute_start,
        interval_minutes=args.interval,
        latitude=args.lat,
        longitude=args.lon,
        timezone_name=args.timezone,
        output_prefix=args.output_prefix,
        expected_mode=args.expected_mode,
        sections = args.section,
    )


if __name__ == "__main__":
    main()
