# app.py
import streamlit as st
import numpy as np
import pandas as pd
import json
import time # Pour l'exemple de chargement

# Importer les classes et fonctions nécessaires depuis votre backend refactorisé
try:
    from backend import (
        MaterialManager, OpticalCalculator,
        calculate_initial_ep, calculate_qwot_from_ep,
        run_local_optimization_workflow, # Ajoutez d'autres workflows si nécessaire
        save_design_json, load_design_json,
        MIN_THICKNESS_PHYS_NM, EXCEL_FILE_PATH # Importez les constantes nécessaires
    )
except ImportError as e:
    st.error(f"Erreur: Impossible d'importer le fichier backend 'backend.py'. Assurez-vous qu'il est dans le même répertoire.\n{e}")
    st.stop() # Arrête l'exécution si le backend n'est pas trouvé

# --- Configuration de la Page Streamlit ---
st.set_page_config(
    page_title="Thin Film Optimizer",
    layout="wide", # Utilise toute la largeur
    initial_sidebar_state="expanded" # Ouvre la sidebar par défaut
)

st.title("🔬 Thin Film Stack Optimizer (Streamlit Version)")

# --- Initialisation et Cache ---

# Mise en cache des ressources coûteuses (Managers)
@st.cache_resource
def get_material_manager(excel_path):
    print("Initialisation MaterialManager...") # Pour le débogage, sera affiché au premier chargement
    try:
        return MaterialManager(excel_file_path=excel_path)
    except FileNotFoundError:
        st.error(f"Fichier Excel '{excel_path}' non trouvé. Vérifiez le chemin et l'emplacement du fichier.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de MaterialManager: {e}")
        st.stop()


@st.cache_resource
def get_optical_calculator(_material_manager): # Dépends du manager
    print("Initialisation OpticalCalculator...")
    return OpticalCalculator(material_manager=_material_manager)

# Obtenir les instances des managers
try:
    material_manager = get_material_manager(EXCEL_FILE_PATH)
    calculator = get_optical_calculator(material_manager)
    AVAILABLE_MATERIALS = ["Constant"] + material_manager.get_available_materials_from_excel()
    AVAILABLE_SUBSTRATES = ["Constant", "Fused Silica", "BK7", "D263"] + [m for m in AVAILABLE_MATERIALS if m != "Constant"]
except Exception as e:
    # Les erreurs sont déjà gérées dans les fonctions get_*, mais st.stop() a pu être appelé.
    st.error(f"Erreur critique lors de l'initialisation du backend: {e}")
    st.stop()


# Initialisation de l'état de session (s'il n'existe pas)
# C'est ici que l'on stocke les paramètres de la conception actuelle
if 'design_state' not in st.session_state:
    print("Initialisation de st.session_state['design_state']")
    st.session_state.design_state = {
        # Paramètres par défaut
        'nH_material': "Nb2O5-Helios" if "Nb2O5-Helios" in AVAILABLE_MATERIALS else "Constant",
        'nL_material': "SiO2-Helios" if "SiO2-Helios" in AVAILABLE_MATERIALS else "Constant",
        'nSub_material': "Fused Silica" if "Fused Silica" in AVAILABLE_SUBSTRATES else "Constant",
        'nH_r': 2.35, 'nH_i': 0.0,
        'nL_r': 1.46, 'nL_i': 0.0,
        'nSub': 1.52,
        'l0': 550.0,
        'emp_str': "1,1,1,1,1", # QWOT nominal
        'initial_layer_number': 5, # Doit correspondre à emp_str initial
        'l_step': 10.0,
        'maxiter': 50, # Valeurs plus faibles pour l'exemple
        'maxfun': 75,
        'targets': [ # Liste de dictionnaires pour les cibles
            {'active': True, 'min': 450.0, 'max': 500.0, 'target_min': 0.0, 'target_max': 0.0},
            {'active': True, 'min': 540.0, 'max': 560.0, 'target_min': 1.0, 'target_max': 1.0},
            {'active': True, 'min': 600.0, 'max': 700.0, 'target_min': 0.0, 'target_max': 0.0},
        ],
        # Résultats de la dernière opération
        'last_result_ep': None,
        'last_result_mat_seq': None,
        'last_result_cost': None,
        'last_result_message': "Prêt.",
        'last_spectrum': None, # {'l': array, 'Ts': array}
        'status_message': "Interface initialisée.",
        'logs': ["Interface initialisée."]
    }

# Helper pour ajouter un log
def log_message(msg):
    timestamp = time.strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    st.session_state.design_state['logs'].insert(0, full_msg) # Ajoute au début
    # Limiter la taille du log si nécessaire
    max_log_entries = 200
    if len(st.session_state.design_state['logs']) > max_log_entries:
        st.session_state.design_state['logs'] = st.session_state.design_state['logs'][:max_log_entries]
    # Optionnel: afficher immédiatement un message court
    # st.toast(msg, icon="ℹ️")


# --- Barre Latérale (Inputs) ---
with st.sidebar:
    st.header("Paramètres de Conception")

    # --- Section Matériaux ---
    with st.expander("Matériaux", expanded=True):
        st.session_state.design_state['nH_material'] = st.selectbox(
            "Matériau H", AVAILABLE_MATERIALS,
            index=AVAILABLE_MATERIALS.index(st.session_state.design_state['nH_material']) if st.session_state.design_state['nH_material'] in AVAILABLE_MATERIALS else 0,
            key='sel_nH'
        )
        if st.session_state.design_state['nH_material'] == "Constant":
            col1, col2 = st.columns(2)
            st.session_state.design_state['nH_r'] = col1.number_input("n'_H", value=st.session_state.design_state['nH_r'], format="%.4f", step=0.01, key='const_nH_r')
            st.session_state.design_state['nH_i'] = col2.number_input("k_H", value=st.session_state.design_state['nH_i'], min_value=0.0, format="%.4f", step=0.001, key='const_nH_i')

        st.session_state.design_state['nL_material'] = st.selectbox(
            "Matériau L", AVAILABLE_MATERIALS,
            index=AVAILABLE_MATERIALS.index(st.session_state.design_state['nL_material']) if st.session_state.design_state['nL_material'] in AVAILABLE_MATERIALS else 0,
            key='sel_nL'
        )
        if st.session_state.design_state['nL_material'] == "Constant":
            col1, col2 = st.columns(2)
            st.session_state.design_state['nL_r'] = col1.number_input("n'_L", value=st.session_state.design_state['nL_r'], format="%.4f", step=0.01, key='const_nL_r')
            st.session_state.design_state['nL_i'] = col2.number_input("k_L", value=st.session_state.design_state['nL_i'], min_value=0.0, format="%.4f", step=0.001, key='const_nL_i')

        st.session_state.design_state['nSub_material'] = st.selectbox(
            "Substrat", AVAILABLE_SUBSTRATES,
            index=AVAILABLE_SUBSTRATES.index(st.session_state.design_state['nSub_material']) if st.session_state.design_state['nSub_material'] in AVAILABLE_SUBSTRATES else 0,
            key='sel_nSub'
        )
        if st.session_state.design_state['nSub_material'] == "Constant":
            st.session_state.design_state['nSub'] = st.number_input("n'_Sub", value=st.session_state.design_state['nSub'], format="%.4f", step=0.01, key='const_nSub')

    # --- Section Stack ---
    with st.expander("Stack Nominal", expanded=True):
        st.session_state.design_state['l0'] = st.number_input(
            "λ₀ Centrage QWOT (nm)", value=st.session_state.design_state['l0'], min_value=1.0, step=10.0, key='l0_input'
        )
        st.session_state.design_state['emp_str'] = st.text_area(
            "QWOT Nominaux (ex: 1,1,0.5,1)", value=st.session_state.design_state['emp_str'], height=100, key='qwot_input'
        )
        try:
            current_qwot_layers = len([float(x) for x in st.session_state.design_state['emp_str'].split(',') if x.strip()])
            st.caption(f"Nombre de couches nominales: {current_qwot_layers}")
        except:
            st.caption("Nombre de couches nominales: Erreur de format QWOT")

    # --- Section Cibles Spectrales ---
    with st.expander("Cibles Spectrales (Transmittance)", expanded=True):
        st.caption("Définir les zones pour l'optimisation.")
        # Utiliser st.data_editor pour une interface tabulaire modifiable
        edited_targets_df = st.data_editor(
            pd.DataFrame(st.session_state.design_state['targets']), # Convertir la liste de dict en DataFrame
            num_rows="dynamic", # Permet d'ajouter/supprimer des lignes
            column_config={ # Configurer les types et limites des colonnes
                "active": st.column_config.CheckboxColumn("Active", default=True),
                "min": st.column_config.NumberColumn("λ min (nm)", min_value=1, step=10.0, format="%.1f"),
                "max": st.column_config.NumberColumn("λ max (nm)", min_value=1, step=10.0, format="%.1f"),
                "target_min": st.column_config.NumberColumn("T @ λmin", min_value=0.0, max_value=1.0, step=0.05, format="%.3f"),
                "target_max": st.column_config.NumberColumn("T @ λmax", min_value=0.0, max_value=1.0, step=0.05, format="%.3f"),
            },
            key='targets_editor'
        )
        # Mettre à jour l'état de session depuis le data_editor
        st.session_state.design_state['targets'] = edited_targets_df.to_dict('records')

        # Afficher le nombre de points pour l'optimisation
        st.session_state.design_state['l_step'] = st.number_input(
            "Pas λ pour calcul/optim (nm)", value=st.session_state.design_state['l_step'], min_value=0.1, step=1.0, key='lambda_step_input'
        )
        try:
            active_targets_list = [t for t in st.session_state.design_state['targets'] if t.get('active', False)]
            if active_targets_list:
                 l_min_optim = min(t['min'] for t in active_targets_list if t.get('min') is not None)
                 l_max_optim = max(t['max'] for t in active_targets_list if t.get('max') is not None)
                 l_step = st.session_state.design_state['l_step']
                 if l_min_optim is not None and l_max_optim is not None and l_step > 0 and l_max_optim >= l_min_optim:
                     num_pts = max(2, int(np.round((l_max_optim - l_min_optim) / l_step)) + 1)
                     st.caption(f"Points pour l'optimisation ≈ {num_pts} (Plage: {l_min_optim:.1f}-{l_max_optim:.1f} nm)")
                 else:
                     st.caption("Plage d'optimisation invalide.")
            else:
                st.caption("Aucune cible active pour l'optimisation.")
        except Exception as e:
            st.caption(f"Erreur calcul points optim: {e}")


    # --- Section Paramètres d'Optimisation ---
    with st.expander("Paramètres d'Optimisation"):
        st.session_state.design_state['maxiter'] = st.number_input(
            "Max Iterations (Opt)", value=st.session_state.design_state['maxiter'], min_value=1, step=10, key='maxiter_input'
        )
        st.session_state.design_state['maxfun'] = st.number_input(
             "Max Evals Fonction (Opt)", value=st.session_state.design_state['maxfun'], min_value=1, step=10, key='maxfun_input'
        )
        # Ajouter d'autres paramètres si nécessaire (seuil thin layer, etc.)

    # --- Section Actions ---
    st.header("Actions")

    col1, col2 = st.columns(2)

    # Bouton Evaluer Nominal
    if col1.button("📊 Evaluer Nominal", key="eval_button", use_container_width=True):
        log_message("Début évaluation nominale...")
        st.session_state.design_state['status_message'] = "Calcul nominal en cours..."
        st.session_state.design_state['last_result_ep'] = None # Efface le précédent état optimisé visuellement

        try:
            # 1. Récupérer les paramètres
            l0 = st.session_state.design_state['l0']
            emp_str = st.session_state.design_state['emp_str']
            nH_mat_id = st.session_state.design_state['nH_material']
            nL_mat_id = st.session_state.design_state['nL_material']
            nSub_mat_id = st.session_state.design_state['nSub_material']
            nInc_mat_id = "AIR" # Simplification pour l'instant

            # Gérer les matériaux constants
            if nH_mat_id == "Constant": nH_mat_id = complex(st.session_state.design_state['nH_r'], st.session_state.design_state['nH_i'])
            if nL_mat_id == "Constant": nL_mat_id = complex(st.session_state.design_state['nL_r'], st.session_state.design_state['nL_i'])
            if nSub_mat_id == "Constant": nSub_mat_id = complex(st.session_state.design_state['nSub'], 0.0)

            # 2. Calculer le vecteur d'épaisseurs nominal
            multipliers = [float(x) for x in emp_str.split(',') if x.strip()]
            num_layers = len(multipliers)
            nominal_mat_seq = [nH_mat_id if i % 2 == 0 else nL_mat_id for i in range(num_layers)]
            nominal_ep = calculate_initial_ep(material_manager, multipliers, l0, nominal_mat_seq)
            st.session_state.design_state['last_result_ep'] = nominal_ep.tolist() # Sauvegarder même si nominal
            st.session_state.design_state['last_result_mat_seq'] = nominal_mat_seq

            # 3. Calculer le spectre
            targets = st.session_state.design_state['targets']
            active_targets_list = [t for t in targets if t.get('active', False)]
            if active_targets_list:
                l_min_plot = min(t['min'] for t in active_targets_list if t.get('min') is not None)
                l_max_plot = max(t['max'] for t in active_targets_list if t.get('max') is not None)
                l_vec_plot_np = np.linspace(l_min_plot, l_max_plot, 401) # Grille fine pour le tracé
            else:
                l_vec_plot_np = np.linspace(400, 800, 401) # Grille par défaut si pas de cible active

            spectrum = calculator.calculate_T_spectrum(
                ep_vector=nominal_ep, material_sequence=nominal_mat_seq,
                nSub_material=nSub_mat_id, l_vec=l_vec_plot_np, nInc_material=nInc_mat_id
            )
            st.session_state.design_state['last_spectrum'] = spectrum
            st.session_state.design_state['status_message'] = f"Evaluation nominale terminée ({len(nominal_ep)} couches)."
            log_message(st.session_state.design_state['status_message'])

        except Exception as e:
            error_msg = f"Erreur évaluation nominale: {e}"
            log_message(error_msg)
            st.session_state.design_state['status_message'] = error_msg
            st.sidebar.error(error_msg)
            st.session_state.design_state['last_spectrum'] = None

        st.rerun() # Force la réexécution pour afficher les résultats

    # Bouton Optimisation Locale
    if col2.button("🚀 Optimiser (Local)", key="optimize_button", use_container_width=True):
        log_message("Début optimisation locale...")
        st.session_state.design_state['status_message'] = "Optimisation locale en cours..."

        with st.spinner("Optimisation en cours..."): # Affiche une icône de chargement
            try:
                # 1. Récupérer les paramètres et état initial
                l0 = st.session_state.design_state['l0']
                emp_str = st.session_state.design_state['emp_str']
                nH_mat_id = st.session_state.design_state['nH_material']
                nL_mat_id = st.session_state.design_state['nL_material']
                nSub_mat_id = st.session_state.design_state['nSub_material']
                nInc_mat_id = "AIR"

                # Gérer matériaux constants
                if nH_mat_id == "Constant": nH_mat_id = complex(st.session_state.design_state['nH_r'], st.session_state.design_state['nH_i'])
                if nL_mat_id == "Constant": nL_mat_id = complex(st.session_state.design_state['nL_r'], st.session_state.design_state['nL_i'])
                if nSub_mat_id == "Constant": nSub_mat_id = complex(st.session_state.design_state['nSub'], 0.0)

                # Utiliser le dernier état optimisé s'il existe, sinon calculer depuis nominal
                if st.session_state.design_state.get('last_result_ep') and st.session_state.design_state.get('last_result_mat_seq'):
                     start_ep = np.array(st.session_state.design_state['last_result_ep'])
                     start_mat_seq = st.session_state.design_state['last_result_mat_seq']
                     log_message("Optimisation démarrée depuis le dernier résultat.")
                else:
                    multipliers = [float(x) for x in emp_str.split(',') if x.strip()]
                    num_layers = len(multipliers)
                    start_mat_seq = [nH_mat_id if i % 2 == 0 else nL_mat_id for i in range(num_layers)]
                    start_ep = calculate_initial_ep(material_manager, multipliers, l0, start_mat_seq)
                    log_message("Optimisation démarrée depuis le QWOT nominal.")

                initial_design_state = {
                    'ep_vector': start_ep,
                    'material_sequence': start_mat_seq,
                    'nSub_material': nSub_mat_id,
                    'nInc_material': nInc_mat_id
                }

                # 2. Préparer les cibles et paramètres pour le backend
                targets = st.session_state.design_state['targets']
                active_targets_list = [t for t in targets if t.get('active', False) and all(t.get(k) is not None for k in ['min','max','target_min','target_max'])]
                if not active_targets_list:
                    raise ValueError("Aucune cible active valide définie pour l'optimisation.")

                l_step = st.session_state.design_state['l_step']
                l_min_optim = min(t['min'] for t in active_targets_list)
                l_max_optim = max(t['max'] for t in active_targets_list)
                num_pts = max(2, int(np.round((l_max_optim - l_min_optim) / l_step)) + 1)
                l_vec_optim_np = np.linspace(l_min_optim, l_max_optim, num_pts) # Utiliser linspace pour cet exemple

                optimizer_options = {
                    'maxiter': st.session_state.design_state['maxiter'],
                    'maxfun': st.session_state.design_state['maxfun'],
                    'disp': False, # Pas d'affichage console depuis scipy
                    'ftol': 1e-9, # Tolérances (peuvent être ajoutées à l'UI)
                    'gtol': 1e-6
                }

                # 3. Appeler le workflow backend
                opt_results = run_local_optimization_workflow(
                    initial_design=initial_design_state,
                    material_manager=material_manager,
                    optical_calculator=calculator,
                    active_targets=active_targets_list,
                    l_vec_optim_np=l_vec_optim_np,
                    optimizer_options=optimizer_options
                )

                # 4. Mettre à jour l'état de session avec les résultats
                st.session_state.design_state['last_result_ep'] = opt_results.get('ep_vector')
                st.session_state.design_state['last_result_mat_seq'] = opt_results.get('material_sequence')
                st.session_state.design_state['last_result_cost'] = opt_results.get('final_cost')
                st.session_state.design_state['last_result_message'] = opt_results.get('message')
                st.session_state.design_state['logs'].extend(opt_results.get('logs', []))
                st.session_state.design_state['status_message'] = opt_results.get('message', "Optimisation terminée avec message inconnu.")

                # 5. Recalculer le spectre pour l'affichage
                if opt_results['success'] and opt_results.get('ep_vector'):
                    l_vec_plot_np = np.linspace(l_min_optim, l_max_optim, 401) # Grille fine pour tracé
                    final_spectrum = calculator.calculate_T_spectrum(
                        ep_vector=opt_results['ep_vector'],
                        material_sequence=opt_results['material_sequence'],
                        nSub_material=opt_results['nSub_material'],
                        l_vec=l_vec_plot_np,
                        nInc_material=opt_results['nInc_material']
                    )
                    st.session_state.design_state['last_spectrum'] = final_spectrum
                else:
                    st.session_state.design_state['last_spectrum'] = None # Pas de spectre si échec

                log_message(st.session_state.design_state['status_message'])

            except Exception as e:
                error_msg = f"Erreur optimisation locale: {e}"
                log_message(error_msg)
                st.session_state.design_state['status_message'] = error_msg
                st.sidebar.error(error_msg)
                st.session_state.design_state['last_spectrum'] = None
                # Garder l'ancien état EP si l'optimisation échoue ? A définir.
                # st.session_state.design_state['last_result_ep'] = None
                # st.session_state.design_state['last_result_mat_seq'] = None

        st.rerun() # Force la réexécution pour afficher les résultats

    # Ajouter d'autres boutons pour les workflows Auto Mode, QWOT Scan ici...


# --- Zone Principale (Affichage) ---

st.header("Résultats")

# Affichage du statut
status_msg = st.session_state.design_state.get('status_message', "Prêt.")
if "erreur" in status_msg.lower() or "error" in status_msg.lower() or "failed" in status_msg.lower():
    st.error(status_msg)
else:
    st.success(status_msg)

# Affichage des métriques clés
col1, col2, col3 = st.columns(3)
current_ep = st.session_state.design_state.get('last_result_ep')
num_layers_res = len(current_ep) if current_ep else 0
col1.metric("Nombre de couches", num_layers_res)

cost = st.session_state.design_state.get('last_result_cost')
cost_display = f"{cost:.4e}" if cost is not None and np.isfinite(cost) else "N/A"
col2.metric("Coût final (MSE)", cost_display)

# Affichage du Spectre (si disponible)
st.subheader("Spectre de Transmittance")
spectrum_data = st.session_state.design_state.get('last_spectrum')

if spectrum_data and spectrum_data.get('l') is not None and spectrum_data.get('Ts') is not None:
    if len(spectrum_data['l']) > 0:
        plot_df = pd.DataFrame({
            'Wavelength (nm)': spectrum_data['l'],
            'Transmittance': spectrum_data['Ts']
        }).set_index('Wavelength (nm)')

        # Ajouter les lignes/points cibles au DataFrame pour les tracer
        target_lines = []
        for t in st.session_state.design_state['targets']:
            if t.get('active'):
                target_lines.append({'Wavelength (nm)': t['min'], 'Target': t['target_min'], 'Target Limit': 'Min'})
                target_lines.append({'Wavelength (nm)': t['max'], 'Target': t['target_max'], 'Target Limit': 'Max'})
        if target_lines:
            target_df = pd.DataFrame(target_lines).set_index('Wavelength (nm)')
            # plot_df = pd.concat([plot_df, target_df], axis=1) # Ne fonctionne pas bien pour les lignes
            # Tracer les cibles séparément pour l'instant
            st.line_chart(plot_df['Transmittance']) # Tracer la transmittance calculée

            # Tracer les points cibles (plus simple que les lignes dans st.line_chart)
            st.write("Points Cibles Actifs:")
            target_points_df = target_df.reset_index()[['Wavelength (nm)', 'Target']]
            st.scatter_chart(target_points_df, x='Wavelength (nm)', y='Target', color=["#FF0000"]*len(target_points_df), size=[20]*len(target_points_df))

        else:
             st.line_chart(plot_df) # Tracer juste la transmittance si pas de cibles

    else:
        st.info("Le calcul du spectre n'a retourné aucune donnée.")
else:
    st.info("Aucun spectre à afficher. Lancez une évaluation ou une optimisation.")


# Affichage des épaisseurs (si disponible)
st.subheader("Structure Optimisée")
ep_vector = st.session_state.design_state.get('last_result_ep')
mat_seq = st.session_state.design_state.get('last_result_mat_seq')

if ep_vector and mat_seq and len(ep_vector) == len(mat_seq):
    layer_data = []
    cumulative_thickness = 0
    for i, (thick, mat) in enumerate(zip(ep_vector, mat_seq)):
        # Afficher le matériau de manière lisible
        if isinstance(mat, complex):
            mat_display = f"Const: {mat.real:.3f}{mat.imag:+.3f}j"
        else:
            mat_display = str(mat)
        layer_data.append({
            "Couche": i + 1,
            "Matériau": mat_display,
            "Epaisseur (nm)": f"{thick:.4f}",
            "z cumulé (nm)": f"{cumulative_thickness + thick:.4f}"
        })
        cumulative_thickness += thick
    st.dataframe(pd.DataFrame(layer_data), use_container_width=True)
else:
    st.info("Aucune structure optimisée disponible.")

# Affichage des Logs
with st.expander("Afficher les Logs"):
    log_content = "\n".join(st.session_state.design_state.get('logs', ["Aucun log."]))
    st.text_area("Logs", value=log_content, height=300, disabled=True, key="log_display")
