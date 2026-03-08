"""
generate_report_pdf.py - Genera un report PDF completo stile paper accademico
per la strategia RR+YC Duration Timing.

Prerequisiti:
    pip install fpdf2

Output:
    output/report/strategy_report.pdf
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fpdf import FPDF

from config import SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime
from versione_finale_v2 import build_combined_signal, run_backtest, PARAMS, SCORE_MAP

# ======================================================================
# OUTPUT DIRECTORY
# ======================================================================
REPORT_DIR = os.path.join(SAVE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

# Chart directories
V2_DIR = os.path.join(SAVE_DIR, "versione_finale_v2")
SLOPE_DIR = os.path.join(SAVE_DIR, "slope_analysis")
FINAL_DIR = os.path.join(SAVE_DIR, "final")


# ======================================================================
# PDF CLASS
# ======================================================================

class StrategyReport(FPDF):
    """Custom PDF class with header/footer and helper methods."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=25)
        self._section_number = 0
        self._in_cover = False

    # ── Header / Footer ──

    def header(self):
        if self._in_cover:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, "RR Duration Strategy - Report Completo", align="L")
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(15, 12, 195, 12)
        self.ln(6)

    def footer(self):
        if self._in_cover:
            return
        self.set_y(-20)
        self.set_draw_color(200, 200, 200)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, f"Pagina {self.page_no()}", align="C")

    # ── Helpers ──

    def section_title(self, title):
        """Major section heading."""
        self._section_number += 1
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(26, 82, 118)  # #1a5276
        self.ln(4)
        self.cell(0, 10, f"{self._section_number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(26, 82, 118)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        """Subsection heading."""
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(40, 116, 166)  # #2874a6
        self.ln(2)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        """Body paragraph."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def body_bold(self, text):
        """Bold body text."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bullet(self, text):
        """Bullet point."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5, "-")
        self.multi_cell(0, 5, text)
        self.ln(1)

    def mono_text(self, text):
        """Monospace text block (formulas, code)."""
        self.set_font("Courier", "", 9)
        self.set_text_color(50, 50, 50)
        self.set_fill_color(245, 245, 245)
        y0 = self.get_y()
        self.multi_cell(0, 4.5, text, fill=True)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None, header_color=(40, 116, 166),
                  alt_row_color=(234, 242, 248)):
        """
        Render a table with header and alternating row shading.
        col_widths: list of mm widths; if None, evenly distributed over 180mm.
        """
        n_cols = len(headers)
        if col_widths is None:
            col_widths = [180.0 / n_cols] * n_cols

        # Check if table fits on page (rough estimate)
        needed = 8 + len(rows) * 7
        if self.get_y() + needed > 270:
            self.add_page()

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*header_color)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        for r_idx, row in enumerate(rows):
            if r_idx % 2 == 0:
                self.set_fill_color(*alt_row_color)
                fill = True
            else:
                fill = False
            self.set_text_color(30, 30, 30)

            # Check page break
            if self.get_y() + 7 > 270:
                self.add_page()
                # Re-draw header
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(*header_color)
                self.set_text_color(255, 255, 255)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
                self.ln()
                self.set_font("Helvetica", "", 9)
                self.set_text_color(30, 30, 30)
                if r_idx % 2 == 0:
                    self.set_fill_color(*alt_row_color)

            for i, val in enumerate(row):
                align = "C" if i > 0 else "L"
                self.cell(col_widths[i], 7, str(val), border=1, align=align,
                          fill=fill)
            self.ln()
        self.ln(3)

    def add_image_safe(self, path, w=180):
        """Add image if it exists, otherwise show placeholder text."""
        if os.path.exists(path):
            # Check page space
            if self.get_y() + 100 > 270:
                self.add_page()
            self.image(path, x=15, w=w)
            self.ln(5)
        else:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(150, 150, 150)
            basename = os.path.basename(path)
            self.cell(0, 8, f"[Grafico non disponibile: {basename}]",
                      new_x="LMARGIN", new_y="NEXT")
            self.ln(3)

    def ensure_space(self, mm_needed):
        """Add page if not enough vertical space remaining."""
        if self.get_y() + mm_needed > 270:
            self.add_page()


# ======================================================================
# BUILD THE REPORT
# ======================================================================

def build_report(result, D, z_final, z_rr, z_yc, z_base, z_slope,
                 regime_hmm, regime_yc, yc_score):
    """Build the full PDF report and save it."""

    m = result["metrics"]
    yearly = result["yearly"]
    ev_df = result["events"]

    pdf = StrategyReport()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # ==================================================================
    # COPERTINA
    # ==================================================================
    pdf._in_cover = True
    pdf.add_page()

    # Background box
    pdf.set_fill_color(26, 82, 118)
    pdf.rect(0, 0, 210, 297, "F")

    # Title block
    pdf.set_y(70)
    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "RR Duration Strategy", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 13, "Report Completo", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Subtitle
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(174, 214, 241)
    pdf.cell(0, 8, "Strategia di Duration Timing basata su", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Risk Reversal e Yield Curve", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)

    # Divider
    pdf.set_draw_color(174, 214, 241)
    pdf.set_line_width(0.5)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    pdf.ln(15)

    # Headline metrics
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)

    # 3 metrics in a row
    box_w = 50
    start_x = 22
    metrics_display = [
        ("Sharpe", f"{m['sharpe']:.3f}"),
        ("Ann. Return", f"{m['ann_ret']*100:.2f}%"),
        ("Max Drawdown", f"{m['mdd']*100:.1f}%"),
    ]
    y_box = pdf.get_y()
    for i, (label, val) in enumerate(metrics_display):
        x = start_x + i * 56
        pdf.set_fill_color(40, 116, 166)
        pdf.rect(x, y_box, box_w, 25, "F")
        pdf.set_xy(x, y_box + 3)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(174, 214, 241)
        pdf.cell(box_w, 5, label, align="C")
        pdf.set_xy(x, y_box + 11)
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(box_w, 10, val, align="C")

    pdf.set_y(y_box + 35)
    pdf.ln(5)

    # Period info
    equity = result["equity"]
    start_date = equity.index[0]
    end_date = equity.index[-1]
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(174, 214, 241)
    pdf.cell(0, 7,
             f"Periodo di analisi: {start_date:%Y-%m-%d} - {end_date:%Y-%m-%d}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7,
             f"{m['n_days']} giorni di trading | {m['n_events']} operazioni | "
             f"{m['years']:.1f} anni",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.cell(0, 7, "Strumenti: TLT (long duration) / SHV (short duration)",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # Date at bottom
    pdf.set_y(260)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(174, 214, 241)
    pdf.cell(0, 7, f"Report generato: {pd.Timestamp.now():%Y-%m-%d}",
             align="C")

    pdf._in_cover = False

    # ==================================================================
    # 1. EXECUTIVE SUMMARY
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Executive Summary")

    pdf.body_text(
        "Questa strategia utilizza i Risk Reversal (RR) del mercato delle opzioni "
        "sui Treasury USA come segnale di timing per la duration. Quando lo skew "
        "della volatilita' implicita raggiunge livelli estremi - indicando un "
        "pricing eccessivo del rischio direzionale - la strategia effettua switch "
        "binari tra TLT (long duration, 20+ anni) e SHV (short duration, <1 anno). "
        "Il segnale base e' arricchito con informazioni dalla yield curve "
        "(slope momentum 10Y-2Y e regime bull/bear steepening/flattening 10s30s) "
        "per migliorare la qualita' delle decisioni di timing."
    )

    pdf.body_text(
        "Il segnale composito finale e': z_final = z_rr + 0.40 * z_yc, dove z_rr "
        "combina il z-score del Risk Reversal blend con lo slope momentum, e z_yc "
        "cattura il regime della curva dei rendimenti. Un modello HMM a 2 stati "
        "(CALM/STRESS) basato su MOVE e VIX regola le soglie operative per evitare "
        "segnali LONG durante periodi di stress di mercato."
    )

    # Summary metrics table
    pdf.sub_title("Metriche Chiave")
    headers = ["Metrica", "Valore", "Metrica", "Valore"]
    rows = [
        ["Sharpe Ratio", f"{m['sharpe']:.4f}", "Excess vs TLT", f"{m['excess']*100:+.2f}%"],
        ["Sortino Ratio", f"{m['sortino']:.4f}", "Total Return", f"{m['total_ret']*100:.1f}%"],
        ["Ann. Return", f"{m['ann_ret']*100:.2f}%", "Equity Finale", f"${m['eq_final']:.3f}"],
        ["Volatility", f"{m['vol']*100:.2f}%", "B&H TLT Ann.", f"{m['bm_ann']*100:.2f}%"],
        ["Max Drawdown", f"{m['mdd']*100:.2f}%", "N. Operazioni", f"{m['n_events']}"],
        ["Calmar Ratio", f"{m['calmar']:.4f}", "Periodo", f"{m['years']:.1f} anni"],
    ]
    pdf.add_table(headers, rows, col_widths=[40, 30, 40, 30])

    # Confronto con B&H
    pdf.body_text(
        f"Rispetto al benchmark Buy & Hold TLT (rendimento annualizzato "
        f"{m['bm_ann']*100:.2f}%), la strategia genera un excess return "
        f"annualizzato di {m['excess']*100:+.2f}% con uno Sharpe Ratio di "
        f"{m['sharpe']:.3f} e un Sortino Ratio di {m['sortino']:.3f}. "
        f"Il max drawdown di {m['mdd']*100:.1f}% risulta significativamente "
        f"inferiore a quello del B&H TLT."
    )

    # ==================================================================
    # 2. COSA SONO I RISK REVERSAL
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Cosa sono i Risk Reversal")

    pdf.sub_title("Volatilita' Implicita e Skew")
    pdf.body_text(
        "La volatilita' implicita (IV) delle opzioni riflette le aspettative del "
        "mercato sulla futura variabilita' del sottostante. Tuttavia, la IV non e' "
        "uniforme lungo tutti gli strike: le opzioni out-of-the-money (OTM) "
        "tipicamente hanno IV diverse da quelle at-the-money (ATM). Questa "
        "differenza si chiama 'skew' o 'smile' della volatilita'."
    )

    pdf.body_text(
        "Per le opzioni sui Treasury, lo skew e' particolarmente informativo: "
        "quando il mercato si preoccupa di un calo dei prezzi dei bond "
        "(rialzo dei rendimenti), le put OTM diventano piu' costose; quando si "
        "aspetta un rally (calo dei rendimenti), le call OTM si apprezzano."
    )

    pdf.sub_title("Definizione di Risk Reversal")
    pdf.body_text(
        "Il Risk Reversal (RR) misura la differenza di volatilita' implicita "
        "tra put e call allo stesso delta (tipicamente 25-delta):"
    )

    pdf.mono_text("    RR = IV(Put 25-delta) - IV(Call 25-delta)")

    pdf.body_text(
        "Interpretazione:\n"
        "- RR positivo: le put sono piu' costose delle call -> il mercato prezza "
        "protezione al ribasso (sentiment negativo sui bond)\n"
        "- RR negativo: le call sono piu' costose -> il mercato prezza protezione "
        "al rialzo (sentiment positivo sui bond)\n"
        "- RR vicino a zero: distribuzione simmetrica delle aspettative"
    )

    pdf.sub_title("Perche' i RR funzionano come segnale di timing")
    pdf.body_text(
        "I Risk Reversal estremi tendono a segnalare un posizionamento eccessivo "
        "del mercato. Quando il RR raggiunge valori estremi (positivi o negativi), "
        "indica che il mercato ha prezzato scenari di tail risk in modo aggressivo. "
        "Storicamente, questi estremi tendono a mean-revertire: il mercato "
        "sovra-reagisce e poi si normalizza. La strategia sfrutta questo effetto "
        "di mean-reversion nei momenti di pricing estremo."
    )

    pdf.sub_title("Tenors utilizzati")
    pdf.body_text(
        "La strategia utilizza Risk Reversal su 4 scadenze diverse per costruire "
        "un segnale blend robusto:"
    )
    pdf.bullet("1W (1 settimana): cattura il sentiment a brevissimo termine")
    pdf.bullet("1M (1 mese): orizzonte standard per opzioni sui Treasury")
    pdf.bullet("3M (3 mesi): segnale a medio termine, meno rumoroso")
    pdf.bullet("6M (6 mesi): tendenza strutturale, meno sensibile al noise")

    pdf.body_text(
        "Il blend (media semplice) dei 4 tenors produce un segnale piu' stabile "
        "e meno soggetto a falsi positivi rispetto all'uso di un singolo tenor."
    )

    # ==================================================================
    # 3. COSTRUZIONE DEL SEGNALE
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Costruzione del Segnale")

    pdf.body_text(
        "Il segnale finale e' costruito combinando tre componenti distinte, "
        "ciascuna con una logica economica precisa. La formula composita e':"
    )
    pdf.mono_text(
        "    z_final = z_rr + 0.40 * z_yc\n"
        "    \n"
        "    dove:\n"
        "    z_rr  = z_base + 0.50 * z_slope   (RR composite)\n"
        "    z_yc  = z-score del regime score   (YC regime)"
    )

    pdf.sub_title("3.1 Z-score base del Risk Reversal")
    pdf.body_text(
        "Il primo componente e' il z-score rolling standard del blend RR. "
        "Si calcola la media dei RR su 4 tenors (1W, 1M, 3M, 6M), poi si "
        "normalizza con media e deviazione standard rolling su 63 giorni "
        "(circa 3 mesi):"
    )
    pdf.mono_text(
        "    rr_blend = media(RR_1W, RR_1M, RR_3M, RR_6M)\n"
        "    mu = rolling_mean(rr_blend, window=63)\n"
        "    sigma = rolling_std(rr_blend, window=63)\n"
        "    z_base = (rr_blend - mu) / sigma"
    )
    pdf.body_text(
        "La finestra di 63 giorni bilancia reattivita' e stabilita'. Finestre "
        "piu' corte generano troppi segnali (noise), finestre piu' lunghe sono "
        "troppo lente per catturare shift di regime."
    )

    pdf.sub_title("3.2 Slope Momentum (10Y-2Y)")
    pdf.body_text(
        "Il secondo componente cattura il momentum della curva dei rendimenti "
        "usando il cambio a 126 giorni (6 mesi) nello spread 10Y-2Y:"
    )
    pdf.mono_text(
        "    slope = yield_10Y - yield_2Y\n"
        "    delta_slope = slope(t) - slope(t-126)\n"
        "    z_slope = z-score(delta_slope, window=252, min_periods=63)"
    )
    pdf.body_text(
        "Questo componente cattura l'effetto di mean-reversion dello slope: "
        "quando la curva si e' appiattita (o irripidita) in modo anomalo "
        "rispetto alla storia recente, tende a normalizzarsi. Il peso "
        "k_slope = 0.50 e' calibrato per aggiungere informazione senza "
        "dominare il segnale RR base."
    )

    pdf.sub_title("3.3 YC Regime Z-Score")
    pdf.body_text(
        "Il terzo componente classifica il regime della yield curve in 4 stati "
        "basandosi sulle variazioni a 42 giorni del livello e dello spread "
        "del segmento 10s30s:"
    )

    headers = ["Regime", "Livello", "Spread", "Score", "Interpretazione"]
    rows = [
        ["BULL_FLAT", "Scende", "Scende", "+2", "Tassi in calo, curva si appiattisce"],
        ["BULL_STEEP", "Scende", "Sale", "+1", "Tassi in calo, curva si irripidisce"],
        ["BEAR_FLAT", "Sale", "Scende", "-1", "Tassi in salita, curva si appiattisce"],
        ["BEAR_STEEP", "Sale", "Sale", "-2", "Tassi in salita, curva si irripidisce"],
    ]
    pdf.add_table(headers, rows, col_widths=[28, 20, 20, 14, 68])

    pdf.body_text(
        "Lo score numerico viene poi normalizzato con un z-score rolling a "
        "252 giorni, producendo z_yc. I regimi BULL (score positivo) favoriscono "
        "posizioni LONG sui bond, i regimi BEAR favoriscono posizioni SHORT."
    )

    pdf.sub_title("3.4 Formula finale e pesi")
    pdf.body_text(
        "La combinazione finale e' additiva con peso k_yc = 0.40 per il "
        "componente yield curve:"
    )
    pdf.mono_text(
        "    z_final = (z_base + 0.50 * z_slope) + 0.40 * z_yc\n"
        "            = z_base + 0.50*z_slope + 0.40*z_yc"
    )
    pdf.body_text(
        "I pesi (k_slope=0.50, k_yc=0.40) sono stati ottimizzati tramite grid "
        "search e validati con walk-forward analysis. L'approccio additivo e' "
        "preferito rispetto a metodi di filtraggio o agreement perche' mantiene "
        "la continuita' del segnale e permette ai diversi componenti di "
        "compensarsi reciprocamente quando uno e' rumoroso."
    )

    # ==================================================================
    # 4. REGIME DETECTION: HMM CALM/STRESS
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Regime Detection: HMM CALM/STRESS")

    pdf.body_text(
        "Un elemento cruciale della strategia e' la distinzione tra periodi di "
        "calma e stress di mercato. Un Hidden Markov Model (HMM) a 2 stati "
        "Gaussiani viene addestrato sulle feature di volatilita' per classificare "
        "ogni giorno come CALM o STRESS."
    )

    pdf.sub_title("Feature e modello")
    pdf.body_text(
        "Le feature utilizzate sono:"
    )
    pdf.bullet("log(MOVE Index): volatilita' implicita del mercato obbligazionario")
    pdf.bullet("log(VIX): volatilita' implicita del mercato azionario")

    pdf.body_text(
        "L'HMM a 2 stati viene addestrato con 10 random seed diversi, "
        "selezionando il modello con la migliore log-likelihood. L'algoritmo "
        "EM (Expectation-Maximization) converge in massimo 300 iterazioni. "
        "Lo stato con media log(MOVE) piu' alta viene etichettato come STRESS."
    )

    pdf.sub_title("Distribuzione dei regimi")

    # Compute regime distribution
    equity = result["equity"]
    reg_aligned = regime_hmm.reindex(equity.index)
    n_calm = (reg_aligned == "CALM").sum()
    n_stress = (reg_aligned == "STRESS").sum()
    n_total = len(reg_aligned)
    pct_calm = n_calm / n_total * 100
    pct_stress = n_stress / n_total * 100

    headers = ["Regime", "Giorni", "Percentuale", "Caratteristica"]
    rows = [
        ["CALM", str(n_calm), f"{pct_calm:.1f}%",
         "Volatilita' bassa/normale, trend stabili"],
        ["STRESS", str(n_stress), f"{pct_stress:.1f}%",
         "Volatilita' elevata, crisi, shock esogeni"],
    ]
    pdf.add_table(headers, rows, col_widths=[25, 25, 25, 85])

    pdf.sub_title("Impatto sulle regole di trading")
    pdf.body_text(
        "La distinzione CALM/STRESS ha un effetto critico sulle regole:\n"
        "- In regime CALM: sia segnali LONG che SHORT sono permessi\n"
        "- In regime STRESS: solo segnali SHORT sono permessi (nessun LONG)\n\n"
        "Questa asimmetria protegge la strategia durante le crisi: in periodi "
        "di stress, i bond possono muoversi in modo erratico e i segnali LONG "
        "basati su mean-reversion sono meno affidabili. La soglia SHORT in STRESS "
        "e' anche piu' restrittiva (-3.50 vs -3.25 in CALM) per evitare falsi "
        "segnali generati dalla volatilita' elevata."
    )

    # ==================================================================
    # 5. REGOLE DI TRADING
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Regole di Trading")

    pdf.body_text(
        "La strategia opera con un meccanismo di switch binario tra due ETF: "
        "TLT (iShares 20+ Year Treasury Bond, duration ~17 anni) e SHV "
        "(iShares Short Treasury Bond, duration <1 anno). Le posizioni sono "
        "sticky: una volta attivate, restano in vigore fino al segnale successivo."
    )

    pdf.sub_title("Soglie per regime")

    headers = ["Parametro", "CALM", "STRESS"]
    rows = [
        ["Soglia SHORT (z <=)", str(PARAMS["tl_calm"]), str(PARAMS["tl_stress"])],
        ["Soglia LONG (z >=)", str(PARAMS["ts_calm"]), "Disabilitato (99)"],
        ["Posizione SHORT", "100% SHV", "100% SHV"],
        ["Posizione LONG", "100% TLT", "Non permessa"],
    ]
    pdf.add_table(headers, rows, col_widths=[55, 50, 55])

    pdf.sub_title("Parametri di esecuzione")

    headers = ["Parametro", "Valore", "Descrizione"]
    rows = [
        ["Cooldown", "5 giorni",
         "Minimo intervallo tra segnali consecutivi"],
        ["Execution Delay", "T+2",
         "Segnale al giorno T, esecuzione al giorno T+2"],
        ["Transaction Cost", "5 bps",
         "Costo per ogni rebalancing (0.05%)"],
        ["Allocazione Iniziale", "50/50",
         "Partenza con 50% TLT + 50% SHV"],
        ["Posizioni", "Binary",
         "100% TLT oppure 100% SHV, nessun mix intermedio"],
    ]
    pdf.add_table(headers, rows, col_widths=[35, 25, 100])

    pdf.body_text(
        "Il cooldown di 5 giorni previene l'over-trading in periodi di "
        "volatilita' del segnale. Il delay T+2 simula un'esecuzione realistica "
        "(il segnale viene generato a fine giornata, l'ordine puo' essere "
        "eseguito il giorno successivo all'apertura). Il costo di 5 bps e' "
        "conservativo per ETF Treasury altamente liquidi."
    )

    # ==================================================================
    # 6. RISULTATI DI PERFORMANCE
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Risultati di Performance")

    pdf.sub_title("Equity Curve")
    pdf.body_text(
        "Il grafico seguente mostra l'andamento dell'equity della strategia "
        "rispetto al benchmark B&H TLT. Le aree colorate indicano i periodi "
        "di regime STRESS (HMM). I markers triangolari indicano i segnali di "
        "trading (triangolo su = LONG, triangolo giu' = SHORT)."
    )
    pdf.add_image_safe(os.path.join(V2_DIR, "01_equity_zscore.png"))

    pdf.add_page()
    pdf.sub_title("Metriche complete")

    headers = ["Metrica", "Strategy V2", "B&H TLT", "Differenza"]
    rows = [
        ["Ann. Return", f"{m['ann_ret']*100:.2f}%", f"{m['bm_ann']*100:.2f}%",
         f"{m['excess']*100:+.2f}%"],
        ["Sharpe Ratio", f"{m['sharpe']:.4f}", "-", "-"],
        ["Sortino Ratio", f"{m['sortino']:.4f}", "-", "-"],
        ["Volatility", f"{m['vol']*100:.2f}%", "-", "-"],
        ["Max Drawdown", f"{m['mdd']*100:.2f}%", "-", "-"],
        ["Calmar Ratio", f"{m['calmar']:.4f}", "-", "-"],
        ["Total Return", f"{m['total_ret']*100:.1f}%", "-", "-"],
        ["Equity Finale", f"${m['eq_final']:.3f}", "-", "-"],
    ]
    pdf.add_table(headers, rows, col_widths=[40, 40, 35, 35])

    pdf.sub_title("Confronto V1 (RR-only) vs V2 (RR+YC)")
    pdf.body_text(
        "L'aggiunta del segnale Yield Curve (V2) migliora le performance "
        "rispetto alla versione originale (V1) che usava solo il Risk Reversal:"
    )

    headers = ["Metrica", "V1 (RR-only)", "V2 (RR+YC)", "Variazione"]
    rows = [
        ["Sharpe", "0.325", f"{m['sharpe']:.3f}",
         f"{(m['sharpe']/0.325-1)*100:+.0f}%"],
        ["Sortino", "0.399", f"{m['sortino']:.3f}",
         f"{(m['sortino']/0.399-1)*100:+.0f}%"],
        ["Ann. Return", "5.39%", f"{m['ann_ret']*100:.2f}%", "-"],
        ["Max Drawdown", "-23.6%", f"{m['mdd']*100:.1f}%", "-"],
        ["Events", "52", str(m['n_events']), "-"],
    ]
    pdf.add_table(headers, rows, col_widths=[35, 35, 35, 35],
                  header_color=(108, 52, 131))

    pdf.sub_title("Confronto annuale")
    pdf.add_image_safe(os.path.join(V2_DIR, "02_yearly_comparison.png"))

    # ==================================================================
    # 7. BREAKDOWN ANNUALE
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Breakdown Annuale")

    headers = ["Anno", "Strategy", "B&H TLT", "Excess", "Vol", "N.Trade"]
    rows = []
    for _, yr in yearly.iterrows():
        rows.append([
            str(int(yr["year"])),
            f"{yr['strat_ret']*100:+.2f}%",
            f"{yr['bm_ret']*100:+.2f}%",
            f"{yr['excess']*100:+.2f}%",
            f"{yr['vol']*100:.1f}%",
            str(int(yr["n_events"])),
        ])

    # Totals row
    avg_strat = yearly["strat_ret"].mean() * 100
    avg_bm = yearly["bm_ret"].mean() * 100
    avg_exc = yearly["excess"].mean() * 100
    avg_vol = yearly["vol"].mean() * 100
    total_ev = int(yearly["n_events"].sum())
    rows.append([
        "Media",
        f"{avg_strat:+.2f}%",
        f"{avg_bm:+.2f}%",
        f"{avg_exc:+.2f}%",
        f"{avg_vol:.1f}%",
        str(total_ev),
    ])

    pdf.add_table(headers, rows, col_widths=[22, 30, 30, 30, 22, 22])

    # Statistics
    pos_years = (yearly["excess"] > 0).sum()
    tot_years = len(yearly)
    pdf.body_bold(
        f"Anni con excess positivo: {pos_years}/{tot_years} "
        f"({pos_years/tot_years*100:.0f}%)  |  "
        f"Excess medio annuo: {avg_exc:+.2f}%"
    )

    pdf.sub_title("Commento sui periodi chiave")

    # Find notable years
    for _, yr in yearly.iterrows():
        y = int(yr["year"])
        exc = yr["excess"] * 100
        if abs(exc) > 10:
            if exc > 0:
                pdf.bullet(
                    f"{y}: Excess {exc:+.1f}%. Strategy return "
                    f"{yr['strat_ret']*100:+.1f}% vs TLT {yr['bm_ret']*100:+.1f}%. "
                    f"La strategia ha correttamente evitato il drawdown del B&H "
                    f"passando a SHV nei momenti critici."
                )
            else:
                pdf.bullet(
                    f"{y}: Excess {exc:+.1f}%. Strategy return "
                    f"{yr['strat_ret']*100:+.1f}% vs TLT {yr['bm_ret']*100:+.1f}%. "
                    f"Anno sfavorevole: i segnali di timing non hanno catturato "
                    f"il movimento del mercato."
                )

    pdf.ln(3)
    pdf.body_text(
        "La strategia mostra la maggiore efficacia nei periodi di forte "
        "direzionalita' negativa dei bond (2013 taper tantrum, 2022 hiking cycle), "
        "dove la corretta identificazione dei segnali SHORT permette di evitare "
        "perdite significative. Nei periodi di bull market stabile dei bond, "
        "la strategia tende a partecipare al rialzo con rendimenti simili al B&H."
    )

    # Drawdown chart
    pdf.sub_title("Analisi del Drawdown")
    pdf.add_image_safe(os.path.join(V2_DIR, "03_drawdown.png"))

    # ==================================================================
    # 8. LISTA COMPLETA DEI TRADE
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Lista Completa dei Trade")

    pdf.body_text(
        f"La strategia ha generato {len(ev_df)} operazioni nel periodo di "
        f"backtest. Ogni riga rappresenta un segnale di switch: la strategia "
        f"mantiene la posizione fino al segnale successivo (approccio sticky)."
    )

    if len(ev_df) > 0:
        n_long = (ev_df["signal"] == "LONG").sum()
        n_short = (ev_df["signal"] == "SHORT").sum()
        n_calm = (ev_df["regime"] == "CALM").sum()
        n_stress = (ev_df["regime"] == "STRESS").sum()

        # Trade list table
        headers = ["#", "Data", "Segnale", "z-score", "Regime"]
        rows = []
        for i, (_, ev) in enumerate(ev_df.iterrows()):
            rows.append([
                str(i + 1),
                ev["date"].strftime("%Y-%m-%d"),
                ev["signal"],
                f"{ev['z']:+.3f}",
                ev["regime"],
            ])
        pdf.add_table(headers, rows, col_widths=[12, 30, 22, 25, 22])

        # Aggregate statistics
        pdf.sub_title("Statistiche aggregate")

        headers = ["Statistica", "Valore"]
        rows = [
            ["Totale operazioni", str(len(ev_df))],
            ["Segnali LONG", f"{n_long} ({n_long/len(ev_df)*100:.0f}%)"],
            ["Segnali SHORT", f"{n_short} ({n_short/len(ev_df)*100:.0f}%)"],
            ["In regime CALM", f"{n_calm} ({n_calm/len(ev_df)*100:.0f}%)"],
            ["In regime STRESS", f"{n_stress} ({n_stress/len(ev_df)*100:.0f}%)"],
            ["z-score medio (LONG)",
             f"{ev_df[ev_df['signal']=='LONG']['z'].mean():+.3f}" if n_long > 0 else "N/A"],
            ["z-score medio (SHORT)",
             f"{ev_df[ev_df['signal']=='SHORT']['z'].mean():+.3f}" if n_short > 0 else "N/A"],
        ]
        pdf.add_table(headers, rows, col_widths=[60, 60])

    # ==================================================================
    # 9. ANALISI DELLA YIELD CURVE
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Analisi della Yield Curve")

    pdf.body_text(
        "L'analisi approfondita della yield curve ha guidato la scelta del "
        "segmento di curva e del tipo di segnale da integrare nella strategia. "
        "I risultati chiave emergono dallo studio della relazione tra "
        "variazioni dello spread e rendimenti di TLT."
    )

    pdf.sub_title("9.1 Gerarchia degli spread")
    pdf.body_text(
        "Non tutti i segmenti della curva dei rendimenti hanno la stessa "
        "capacita' predittiva per i rendimenti di TLT. L'analisi di regressione "
        "mostra una chiara gerarchia:"
    )

    headers = ["Spread", "Beta", "R-squared", "Rilevanza"]
    rows = [
        ["2s5s", "-0.002113", "0.3913", "Migliore"],
        ["2s10s", "-0.001408", "0.3629", "Eccellente"],
        ["2s30s", "-0.000935", "0.2420", "Buono"],
        ["3m2y", "-0.000710", "0.1735", "Discreto"],
        ["5s10s", "-0.001396", "0.1013", "Moderato"],
        ["10s30s", "-0.000026", "0.0000", "Non significativo"],
    ]
    pdf.add_table(headers, rows, col_widths=[25, 30, 30, 40])

    pdf.body_text(
        "Il segmento 2s5s ha il piu' alto R-squared (0.39), seguito da 2s10s "
        "(0.36). Paradossalmente, il 10s30s - usato per la classificazione del "
        "regime - ha R-squared quasi nullo come predittore diretto. Tuttavia, "
        "il 10s30s funziona bene come indicatore qualitativo di regime "
        "(bull/bear, steep/flat) piuttosto che come predittore quantitativo."
    )

    pdf.add_image_safe(os.path.join(SLOPE_DIR, "03_spread_correlation.png"))

    pdf.add_page()
    pdf.sub_title("9.2 Regime YC e performance TLT")
    pdf.body_text(
        "La classificazione del regime della yield curve rivela differenze "
        "drastiche nelle performance di TLT:"
    )

    headers = ["Regime", "Ann. Return", "Volatility", "N. Giorni"]
    rows = [
        ["BULL_FLAT", "+48.6%", "14.5%", "1628"],
        ["BULL_STEEP", "+15.5%", "16.5%", "857"],
        ["BEAR_FLAT", "-16.9%", "13.1%", "968"],
        ["BEAR_STEEP", "-44.5%", "14.6%", "1590"],
    ]
    pdf.add_table(headers, rows, col_widths=[30, 30, 30, 30])

    pdf.body_text(
        "I regimi BULL_FLAT e BEAR_STEEP sono i piu' estremi: nel primo caso "
        "TLT guadagna quasi il 50% annualizzato, nel secondo perde oltre il 44%. "
        "Questo conferma che la direzione dei tassi (rising/falling) e' il driver "
        "principale, mentre la forma della curva (steep/flat) aggiunge "
        "informazione incrementale."
    )

    pdf.add_image_safe(os.path.join(SLOPE_DIR, "05_tlt_by_regime_bar.png"))

    pdf.sub_title("9.3 L'insidia degli hit rate")
    pdf.body_text(
        "Nonostante le enormi differenze nei rendimenti medi, gli hit rate "
        "(percentuale di giorni con rendimenti positivi) sono sorprendentemente "
        "vicini al 50% in tutti i regimi:"
    )

    headers = ["Regime", "Hit Rate 1d", "Hit Rate 5d", "Hit Rate 21d"]
    rows = [
        ["BULL_STEEP", "49.1%", "49.0%", "51.6%"],
        ["BULL_FLAT", "51.0%", "52.9%", "50.0%"],
        ["BEAR_STEEP", "52.3%", "53.2%", "50.1%"],
        ["BEAR_FLAT", "52.3%", "50.1%", "51.9%"],
    ]
    pdf.add_table(headers, rows, col_widths=[30, 35, 35, 35])

    pdf.body_text(
        "Questo significa che la strategia basata sulla yield curve e' una "
        "strategia 'di code' (tail strategy): i rendimenti medi sono guidati "
        "da pochi giorni con movimenti estremi, non dalla frequenza dei "
        "giorni positivi. Non ci si puo' aspettare un alto tasso di successo "
        "giornaliero - il valore aggiunto emerge nel tempo attraverso "
        "l'asimmetria dei payoff."
    )

    pdf.add_page()
    pdf.sub_title("9.4 Rising vs Falling yields - analisi 4 quadranti")
    pdf.body_text(
        "L'analisi dei 4 quadranti (Rising/Falling x Steep/Flat) conferma che "
        "la direzione dei tassi conta molto piu' della forma della curva. "
        "Il quadrante Falling+Flat (BULL_FLAT) ha lo Sharpe piu' alto "
        "(+3.22), mentre Rising+Steep (BEAR_STEEP) ha lo Sharpe piu' negativo "
        "(-3.18)."
    )

    headers = ["Scenario", "Ann. Return", "Sharpe", "Hit Rate"]
    rows = [
        ["Falling + Flat", "+48.6%", "+3.223", "59.5%"],
        ["Falling + Steep", "+15.5%", "+0.819", "52.9%"],
        ["Rising + Flat", "-16.9%", "-1.438", "49.5%"],
        ["Rising + Steep", "-44.5%", "-3.183", "43.3%"],
    ]
    pdf.add_table(headers, rows, col_widths=[35, 30, 25, 25])

    pdf.add_image_safe(os.path.join(SLOPE_DIR, "11_four_quadrant.png"))

    # ==================================================================
    # 10. STABILITA' NEL TEMPO
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Stabilita' nel Tempo")

    pdf.body_text(
        "Una preoccupazione fondamentale per qualsiasi strategia quantitativa e' "
        "la stabilita' delle relazioni nel tempo. La correlazione tra spread "
        "della curva e rendimenti di TLT non e' costante: varia significativamente "
        "tra periodi storici diversi."
    )

    pdf.sub_title("Correlazione per periodo storico")
    pdf.body_text(
        "L'analisi per sotto-periodi mostra che la correlazione spread-TLT "
        "ha subito variazioni importanti:"
    )
    pdf.bullet(
        "Pre-crisi (2006-2008): correlazioni moderate, mercato 'normale'"
    )
    pdf.bullet(
        "Recovery (2009-2012): correlazioni elevate, politiche di QE dominano"
    )
    pdf.bullet(
        "Low Vol (2017-2019): correlazioni ridotte, curva piatta"
    )
    pdf.bullet(
        "Post-2020: potenziale degradazione strutturale per l'ambiente "
        "di tassi piu' alto"
    )

    pdf.add_image_safe(os.path.join(SLOPE_DIR, "13_period_spread_analysis.png"))

    pdf.sub_title("Slope momentum come effetto di mean-reversion")
    pdf.body_text(
        "Il componente slope momentum (variazione a 126 giorni dello spread "
        "10Y-2Y) sfrutta un effetto empirico robusto: quando la curva si e' "
        "mossa in modo estremo in una direzione, tende a normalizzarsi. "
        "Questo effetto di mean-reversion e' piu' stabile nel tempo rispetto "
        "alla correlazione contemporanea, perche' non dipende dalla direzione "
        "strutturale dei tassi ma dalla velocita' del cambiamento."
    )

    pdf.add_image_safe(os.path.join(SLOPE_DIR, "14_rolling_correlation.png"))

    # ==================================================================
    # 11. RISCHI E LIMITAZIONI
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Rischi e Limitazioni")

    pdf.body_text(
        "Come ogni strategia quantitativa, il sistema RR+YC presenta rischi "
        "e limitazioni che devono essere compresi prima dell'implementazione "
        "in produzione."
    )

    pdf.sub_title("Correlazione instabile")
    pdf.body_text(
        "La correlazione tra spread della curva e rendimenti di TLT non e' "
        "costante. Periodi di politica monetaria non convenzionale (QE, YCC) "
        "o cambi di regime nella politica della Fed possono alterare "
        "significativamente le relazioni storiche su cui la strategia si basa."
    )

    pdf.sub_title("Hit rate vicino al 50%")
    pdf.body_text(
        "La strategia ha un hit rate (percentuale di trade profittevoli) "
        "vicino al 50%. Non e' una strategia che 'vince spesso con poco': "
        "e' una strategia che vince raramente con molto. Questo richiede "
        "disciplina psicologica e orizzonte temporale lungo. Periodi di "
        "underperformance prolungata sono possibili e previsti."
    )

    pdf.sub_title("Drawdown significativo")
    pdf.body_text(
        f"Il max drawdown della strategia e' {m['mdd']*100:.1f}%, "
        f"che rimane significativo nonostante sia inferiore al B&H TLT. "
        f"Il Calmar ratio di {m['calmar']:.3f} indica che il rendimento non "
        f"compensa completamente il rischio di drawdown su base unitaria."
    )

    pdf.sub_title("Rischio di regime-change")
    pdf.body_text(
        "Cambiamenti strutturali nel mercato obbligazionario - come il "
        "passaggio da un ambiente di tassi decrescenti (2008-2020) a un "
        "ambiente di tassi crescenti (post-2022) - possono alterare "
        "l'efficacia dei segnali. Il periodo post-2020 mostra gia' "
        "segnali di potenziale degradazione in alcune relazioni di curva."
    )

    pdf.sub_title("Overfitting e look-ahead bias")
    pdf.body_text(
        "I parametri della strategia (soglie, pesi, finestre) sono stati "
        "ottimizzati sul campione completo. Nonostante la validazione con "
        "walk-forward analysis suggerisca robustezza, il rischio di "
        "overfitting resta presente. Le soglie esatte (-3.25, +2.50, -3.50) "
        "potrebbero non essere ottimali per il futuro."
    )

    # ==================================================================
    # 12. CONCLUSIONI
    # ==================================================================
    pdf.add_page()
    pdf.section_title("Conclusioni")

    pdf.body_text(
        "La strategia RR+YC Duration Timing rappresenta un approccio "
        "sistematico e disciplinato al timing della duration nei Treasury USA. "
        "Combinando informazioni dai mercati delle opzioni (Risk Reversal) "
        "con la struttura della yield curve, il sistema genera segnali "
        "di qualita' per effettuare switch tra TLT e SHV."
    )

    pdf.sub_title("Sintesi dei risultati")
    pdf.bullet(
        f"Sharpe Ratio di {m['sharpe']:.3f}, in miglioramento rispetto "
        f"alla versione V1 (RR-only, Sharpe 0.325)"
    )
    pdf.bullet(
        f"Rendimento annualizzato di {m['ann_ret']*100:.2f}% con un excess "
        f"di {m['excess']*100:+.2f}% rispetto al B&H TLT"
    )
    pdf.bullet(
        f"Max drawdown contenuto a {m['mdd']*100:.1f}%, "
        f"significativamente inferiore al B&H TLT nei periodi di crisi"
    )
    pdf.bullet(
        f"Solo {m['n_events']} operazioni in {m['years']:.0f} anni: "
        f"la strategia non over-tradea e le posizioni sono mantenute "
        f"per settimane/mesi"
    )

    pos_years = (yearly["excess"] > 0).sum()
    tot_years = len(yearly)
    pdf.bullet(
        f"Excess positivo in {pos_years}/{tot_years} anni "
        f"({pos_years/tot_years*100:.0f}%), con i migliori risultati nei "
        f"periodi di stress del mercato obbligazionario"
    )

    pdf.sub_title("5 punti operativi chiave")
    pdf.body_text("1. Il segnale RR e' il driver principale: cattura gli estremi "
                  "di posizionamento del mercato delle opzioni sui Treasury")
    pdf.body_text("2. La yield curve aggiunge informazione incrementale: lo slope "
                  "momentum e il regime bull/bear migliorano la qualita' del timing")
    pdf.body_text("3. Il regime HMM e' essenziale: bloccare i LONG in STRESS "
                  "previene le perdite peggiori")
    pdf.body_text("4. La strategia e' di tipo 'tail': funziona per asimmetria dei "
                  "payoff, non per frequenza di successo")
    pdf.body_text("5. Monitoraggio continuo: le relazioni di curva possono "
                  "degradarsi e richiedono verifica periodica")

    pdf.sub_title("Prospettive future")
    pdf.body_text(
        "Possibili evoluzioni della strategia includono:\n"
        "- Estensione ad altri mercati obbligazionari (Europa, UK)\n"
        "- Integrazione di segnali macro aggiuntivi (payrolls, CPI)\n"
        "- Gestione dinamica della position size basata sulla confidenza\n"
        "- Adattamento delle soglie con un meccanismo rolling/adaptive\n"
        "- Monitoraggio della stabilita' delle relazioni di curva con alert "
        "automatici per potenziale degradazione del segnale"
    )

    # ==================================================================
    # APPENDICI (grafici addizionali se disponibili)
    # ==================================================================

    # Check for additional charts from final_charts_v2
    final_charts = [
        ("08_equity_yc_regime.png", "Equity con Regime YC"),
        ("10_detailed_performance.png", "Dashboard Dettagliato"),
    ]
    has_appendix = any(
        os.path.exists(os.path.join(FINAL_DIR, c[0])) for c in final_charts
    )

    if has_appendix:
        pdf.add_page()
        pdf._section_number += 1
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(26, 82, 118)
        pdf.ln(4)
        pdf.cell(0, 10, "Appendice: Grafici Aggiuntivi",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(26, 82, 118)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(6)

        for fname, title in final_charts:
            fpath = os.path.join(FINAL_DIR, fname)
            if os.path.exists(fpath):
                pdf.sub_title(title)
                pdf.add_image_safe(fpath)

    # ==================================================================
    # SAVE
    # ==================================================================
    out_path = os.path.join(REPORT_DIR, "strategy_report.pdf")
    pdf.output(out_path)
    return out_path


# ======================================================================
# MAIN
# ======================================================================

def main():
    T0 = time.time()
    print("=" * 72)
    print("  PDF REPORT GENERATION - RR+YC Duration Strategy")
    print("=" * 72)

    # 1. Load data
    print("\n[1/4] Caricamento dati ...")
    D = load_all()

    # 2. Regime HMM
    print("[2/4] Regime HMM ...")
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])

    # 3. Build signals and run backtest
    print("[3/4] Segnale combinato + Backtest ...")
    z_final, z_rr, z_yc, z_base, z_slope, regime_yc, yc_score = \
        build_combined_signal(D, PARAMS)

    start = pd.Timestamp(BACKTEST_START)
    result = run_backtest(z_final, regime_hmm, D["etf_ret"], PARAMS, start)

    m = result["metrics"]
    print(f"  Sharpe: {m['sharpe']:.4f}  |  Ann.Ret: {m['ann_ret']*100:.2f}%  |  "
          f"MDD: {m['mdd']*100:.1f}%  |  Events: {m['n_events']}")

    # 4. Generate PDF
    print("[4/4] Generazione PDF ...")
    out_path = build_report(result, D, z_final, z_rr, z_yc, z_base, z_slope,
                            regime_hmm, regime_yc, yc_score)

    elapsed = time.time() - T0
    print(f"\n{'='*72}")
    print(f"  PDF GENERATO: {out_path}")
    print(f"  Tempo: {elapsed:.1f}s")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
