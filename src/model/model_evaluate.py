import pandas as pd
from IPython.display import display
from ipywidgets import Output


def evaluate_and_display_model(
    net,
    model_name: str,
    summary: dict,
    test_loader,
    hist_df: pd.DataFrame,
    results_df: pd.DataFrame,
    notes: str = "",
    device: str = "cuda",
    use_cache: dict | None = None,
) -> tuple[pd.DataFrame, object]:
    """
    Führt Evaluation durch, baut ein ipywidgets-Panel auf und gibt
    (results_df, panel) zurück, ohne es selbst zu displayen.
    """
    from src import model, util

    if use_cache is None:
        use_cache = {
            "training": False,
            "confmat": False,
            "scores": False,
            "table": False,
        }

    # ─── 1. Eval auf Testdaten ─────────────────────────────────────────
    y_true, y_prob = model.model_train._collect_probs(net, test_loader, device=device)
    y_pred = model.predict_binary(y_prob)
    test_acc = (y_pred == y_true).mean()
    test_wauc = model.weighted_auc(y_true, y_prob)

    # ─── 2. Ergebnis-Row speichern ─────────────────────────────────────
    result_row = {
        "model_name": model_name,
        "best_epoch": summary["best_epoch"],
        "val_acc": summary["final_val_acc"],
        "val_wauc": summary["final_val_wauc"],
        "test_acc": test_acc,
        "test_wauc": test_wauc,
        "params": sum(p.numel() for p in net.parameters()),
        "notes": notes,
    }
    results_df.loc[len(results_df)] = result_row

    # ─── 3. Confusion & ROC Daten ─────────────────────────────────────
    confmat = model.confusion_counts(y_true, y_pred, binary=True)
    roc_dict = model.roc_data(y_true, y_prob)
    roc_dict["label"] = model_name
    roc_dict["wauc"] = test_wauc

    # ─── 4. Ein einziges Output-Widget für die Tabelle ───────────────
    table_out = Output()
    with table_out:
        display(results_df)

    # ─── Hilfswrapper, die das injizierte df abfangen ───────────────────
    # für Funktionen ohne 'df' Parameter:
    def wrap_no_df(func, /, **fixed_kwargs):
        # erzeugt eine Callable, die df und weitere kwargs schluckt
        def wrapped(df=None, **kwargs):
            # merge fixed kwargs und jene von toggle
            return func(**fixed_kwargs, **kwargs)

        return wrapped

    # ─── 6. Toggle-Helper erzeugen ────────────────────────────────────
    toggle_eval = util.make_toggle_shortcut(hist_df, model_name)

    # ─── 7. Definition der Tabs ───────────────────────────────────────
    eval_training = [toggle_eval("1-1. Lernverlauf", model.plot_history)]
    eval_confmat = [
        toggle_eval(
            "2-1. Konfusionsmatrix",
            # wrap_confmat schluckt df und ruft mit cm und labels
            wrap_no_df(
                model.plot_confmat,
                cm=confmat,
                labels_true=["Cover", "Stego"],
                labels_pred=["Cover", "Stego"],
            ),
        )
    ]
    eval_scores = [
        toggle_eval("3-1. ROC-Kurve", wrap_no_df(model.plot_roc_curves, curves=[roc_dict])),
        toggle_eval("3-2. Score-Verteilung", wrap_no_df(model.plot_score_histogram, y_prob=y_prob, y_true=y_true)),
    ]
    eval_table = [
        # auch hier df wird geschluckt
        toggle_eval("4-1. Ergebnistabelle", wrap_no_df(lambda: table_out))
    ]

    eval_sections = [
        util.make_dropdown_section(eval_training, model_name, use_cache=use_cache["training"]),
        util.make_dropdown_section(eval_confmat, model_name, use_cache=use_cache["confmat"]),
        util.make_dropdown_section(eval_scores, model_name, use_cache=use_cache["scores"]),
        util.make_dropdown_section(eval_table, model_name, use_cache=use_cache["table"]),
    ]

    eval_tabs = [
        "1. Trainingsverlauf",
        "2. Konfusionsmatrix",
        "3. Score-Auswertung",
        "4. Ergebnistabelle",
    ]

    panel = util.make_lazy_panel_with_tabs(
        eval_sections,
        eval_tabs,
        open_btn_text=f"{model_name} Evaluation anzeigen",
        close_btn_text="Schliessen",
    )

    return results_df, panel
