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
    start="2026-06-01 00:00:00",
    interval_minutes=10,
    latitude=None,
    longitude=None,
    timezone_name="Europe/Berlin",
    output_prefix=None,
    expected_mode="auto",
    house_sky_border=120,
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError("Datei nicht gefunden: {}".format(image_path))

    if output_prefix is None:
        output_prefix = image_path.stem + "_helligkeit"

    output_csv = image_path.with_name(output_prefix + ".csv")
    output_plot = image_path.with_name(output_prefix + ".png")

    start_time = parse_start_time(start)

    img = Image.open(image_path).convert("RGB")

    width, height = img.size

    draw_img = img.convert("RGBA")
    # Transparente Overlay-Ebene erstellen
    overlay = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        [(0, 0), (width, house_sky_border)],
        fill=(255, 0, 0, 80)  # Rot mit leichter Transparenz
    )
    draw.rectangle(
        [(0, house_sky_border), (width, height)],
        fill=(0, 255, 0, 80)  # Rot mit leichter Transparenz
    )
    draw_img = Image.alpha_composite(draw_img, overlay)

    rgb_display = np.asarray(img, dtype=np.uint8)
    rgb_float = rgb_display.astype(np.float32)

    luminance_house = (
        0.2126 * rgb_float[:house_sky_border, :, 0]
        + 0.7152 * rgb_float[:house_sky_border, :, 1]
        + 0.0722 * rgb_float[:house_sky_border, :, 2]
    )
    luminance_sky = (
        0.2126 * rgb_float[house_sky_border:, :, 0]
        + 0.7152 * rgb_float[house_sky_border:, :, 1]
        + 0.0722 * rgb_float[house_sky_border:, :, 2]
    )

    house_sum = luminance_house.sum(axis=0)
    house_mean = luminance_house.mean(axis=0)
    sky_sum = luminance_sky.sum(axis=0)
    sky_mean = luminance_sky.mean(axis=0)

    brightness_sum = house_sum + sky_sum
    brightness_mean = house_mean + sky_mean

    timestamps = [
        start_time + timedelta(minutes=i * interval_minutes)
        for i in range(width)
    ]

    end_time = timestamps[-1]

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

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "brightness_sum",
            "brightness_mean",
        ])
        for ts, bsum, bmean in zip(timestamps, brightness_sum, brightness_mean):
            writer.writerow([
                ts.isoformat(sep=" "),
                float(bsum),
                float(bmean),
            ])

    x_num = mdates.date2num(timestamps)
    x_min = x_num[0]
    x_max = x_num[-1]

    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(
        8,
        1,
        figsize=(16, 20),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 2.5, 2, 2, 2, 2, 2, 2]},
    )

    ax0.imshow(
        rgb_display,
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

    ax1.imshow(
        draw_img,
        aspect="auto",
        extent=[x_min, x_max, height, 0],
        interpolation="nearest",
    )
    ax1.set_ylabel("Pixelhöhe")
    ax1.set_title("LongKeogram Bereiche")

    info_text = "Bereich mit Hauswand: Rot\nBereich ohne Hauswand mit Himmel: Grün"
    ax1.text(
        0.01,
        0.98,
        info_text,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    ax2.plot(timestamps, brightness_sum, label="Keogramm: Helligkeit Summe")
    ax2.set_ylabel("Summe")
    ax2.set_title("Bereich: ganzes Keogramm - Helligkeitssumme pro vertikaler Linie")
    ax2.grid(True, alpha=0.3)

    ax3.plot(timestamps, brightness_mean, label="Keogramm: Helligkeit Mittelwert")
    ax3.set_ylabel("Mittelwert")
    ax3.set_xlabel("Zeit")
    ax3.set_title("Bereich: ganzes Keogramm - Helligkeitsmittelwert pro vertikaler Linie")
    ax3.grid(True, alpha=0.3)

    ax4.plot(timestamps, house_sum, label="Hauswand: Helligkeit Summe")
    ax4.set_ylabel("Summe")
    ax4.set_title("Bereich: Hauswand - Helligkeitssumme pro vertikaler Linie")
    ax4.grid(True, alpha=0.3)

    ax5.plot(timestamps, house_mean, label="Hauswand: Helligkeit Mittelwert")
    ax5.set_ylabel("Mittelwert")
    ax5.set_xlabel("Zeit")
    ax5.set_title("Bereich: Hauswand - Helligkeitsmittelwert pro vertikaler Linie")
    ax5.grid(True, alpha=0.3)

    ax6.plot(timestamps, sky_sum, label="Himmel: Helligkeit Summe")
    ax6.set_ylabel("Summe")
    ax6.set_title("Bereich: Himmel - Helligkeitssumme pro vertikaler Linie")
    ax6.grid(True, alpha=0.3)

    ax7.plot(timestamps, sky_mean, label="Himmel: Helligkeit Mittelwert")
    ax7.set_ylabel("Mittelwert")
    ax7.set_xlabel("Zeit")
    ax7.set_title("Bereich: Himmel - Helligkeitsmittelwert pro vertikaler Linie")
    ax7.grid(True, alpha=0.3)

    for sunrise in sunrise_list:
        for ax in (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7):
            ax.axvline(sunrise, color="green", linestyle="--", linewidth=1, alpha=0.9)

    for sunset in sunset_list:
        for ax in (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7):
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

    for ax in (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7):
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    plt.suptitle("LongKeogram-Auswertung", fontsize=14, y=0.99)
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
    parser.add_argument("--start", default="2026-06-01 00:00:00", help="Startzeit, Default: 2026-06-01 00:00:00")
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

    args = parser.parse_args()

    analyse_longkeogram(
        image_path=args.image,
        start=args.start,
        interval_minutes=args.interval,
        latitude=args.lat,
        longitude=args.lon,
        timezone_name=args.timezone,
        output_prefix=args.output_prefix,
        expected_mode=args.expected_mode,
    )


if __name__ == "__main__":
    main()
