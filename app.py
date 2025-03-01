import re
import warnings
from io import BytesIO

import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image
import py3Dmol

# RDKit imports wrapped in a try/except block
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    st.error("RDKit modules could not be fully imported. Ensure RDKit is installed with full drawing support.")
    from rdkit import Chem
    Descriptors = None
    rdMolDescriptors = None
    AllChem = None
    Draw = None
    rdMolDraw2D = None

warnings.filterwarnings("ignore")

##############################
# Streamlit Page & Custom Styling
##############################
st.set_page_config(
    page_title="Dye Color Analysis - Enhanced",
    layout="wide",
    initial_sidebar_state="expanded"
)

def set_custom_css() -> None:
    """Inject custom CSS for a retro-futuristic, art deco look."""
    custom_css = """
    <style>
    /* Overall background and text color */
    body {
        background-color: #1e1e2f;
        color: #f8f8f2;
        font-family: 'Courier New', Courier, monospace;
    }
    /* Sidebar styling */
    .css-1d391kg { 
        background-color: #282a36;
    }
    /* Button styling */
    .stButton>button {
        background-color: #ff79c6;
        color: #282a36;
        font-weight: bold;
    }
    /* DataFrame table cell styling */
    .stDataFrame td, .stDataFrame th {
        background-color: #282a36 !important;
        color: #f8f8f2 !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

set_custom_css()

##############################
# SMARTS Patterns Definitions
##############################
# Chromophore patterns – key substructures that drive color
CHROMOPHORE_PATTERNS = {
    'Azo': Chem.MolFromSmarts('N=N'),
    'Anthraquinone': Chem.MolFromSmarts('O=C1C=CC2=CC=CC=C2C1=O'),
    'Nitro': Chem.MolFromSmarts('[NX3](=O)=O'),
    'Quinone': Chem.MolFromSmarts('O=C1C=CC=CC1=O'),
    'Indigoid': Chem.MolFromSmarts('C1C=CC(=O)NC1=O'),
    'Cyanine': Chem.MolFromSmarts('C=C-C=C'),
    'Xanthene': Chem.MolFromSmarts('O1C=CC2=C1C=CC=C2'),
    'Thiazine': Chem.MolFromSmarts('N1C=NC=S1'),
    'Coumarin': Chem.MolFromSmarts('O=C1OC=CC2=CC=CC=C12'),
    'Porphyrin': Chem.MolFromSmarts('N1C=CC=N1'),
    'Phthalocyanine': Chem.MolFromSmarts('C1=C(C2=NC=C(C)N2)C3=CC=CC=C13'),
    'Carotenoid': Chem.MolFromSmarts('C=C(C)C=CC=C'),
    'Squaraine': Chem.MolFromSmarts('C=CC=C'),
    'Metal Complex': Chem.MolFromSmarts('[!#1]'),
    'Bromine': Chem.MolFromSmarts('Br'),
    'Selenium': Chem.MolFromSmarts('[Se]'),
    'Pyridine': Chem.MolFromSmarts('C1=CC=NC=C1'),
    'Phosphine': Chem.MolFromSmarts('P(C)(C)C'),
    'Carbene': Chem.MolFromSmarts('[C]')
}

# Auxochrome patterns – groups that modulate chromophore intensity/position
AUXOCHROME_PATTERNS = {
    'Hydroxyl': Chem.MolFromSmarts('[OX2H]'),
    'Amine': Chem.MolFromSmarts('N'),
    'Methoxy': Chem.MolFromSmarts('COC'),
    'Thiol': Chem.MolFromSmarts('[SX2H]'),
    'Carboxyl': Chem.MolFromSmarts('C(=O)[OX2H1]')
}

##############################
# Scientific λmax Estimation (Woodward–Fieser-inspired)
##############################
LAMBDA_BASE_VALUES = {
    'Azo': 450,
    'Anthraquinone': 550,
    'Nitro': 350,
    'Quinone': 400,
    'Indigoid': 610,
    'Cyanine': 700,
    'Xanthene': 520,
    'Thiazine': 600,
    'Coumarin': 400,
    'Porphyrin': 420,
    'Phthalocyanine': 680,
    'Carotenoid': 480,
    'Squaraine': 700,
    'Bromine': 300,
    'Selenium': 300,
    'Pyridine': 250,
    'Phosphine': 250,
    'Carbene': 250,
    'Metal Complex': 400
}

def count_ring_fusions(mol: Chem.Mol) -> int:
    """Count fused rings (sharing at least 2 atoms) in a molecule."""
    ring_info = mol.GetRingInfo()
    fusion_count = 0
    rings = ring_info.AtomRings()
    for i in range(len(rings)):
        ring_i = set(rings[i])
        for j in range(i + 1, len(rings)):
            ring_j = set(rings[j])
            if len(ring_i.intersection(ring_j)) >= 2:
                fusion_count += 1
    return fusion_count

def has_strong_auxochrome(mol: Chem.Mol) -> bool:
    """Detect if the molecule contains strong auxochromes."""
    patterns = [
        Chem.MolFromSmarts('[OX2H]'),
        Chem.MolFromSmarts('[NX3;H2,H1]')
    ]
    return any(mol.HasSubstructMatch(p) for p in patterns)

def estimate_lambda_max_scientific(smiles: str, chromophore: str = None) -> int:
    """Estimate the λmax using a Woodward–Fieser-inspired approach."""
    base = LAMBDA_BASE_VALUES.get(chromophore, 200)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return base
    conj_dbs = sum(1 for b in mol.GetBonds() 
                   if b.GetIsConjugated() and b.GetBondType() == Chem.rdchem.BondType.DOUBLE)
    extra_conjugation = max(0, conj_dbs - 2)
    incremental = 30 * extra_conjugation
    fusions = count_ring_fusions(mol)
    incremental += 10 * fusions
    if has_strong_auxochrome(mol):
        incremental += 15
    return base + incremental

def nm_to_color_category(wavelength_nm: int) -> str:
    """Convert estimated wavelength to a human-friendly color category."""
    if wavelength_nm <= 200:
        return "Colorless/UV region"
    elif wavelength_nm < 350:
        return "Likely colorless to very pale (UV/near-UV)"
    elif wavelength_nm < 400:
        return "Near-UV/Violet"
    elif wavelength_nm < 450:
        return "Blue/Violet"
    elif wavelength_nm < 500:
        return "Blue/Green"
    elif wavelength_nm < 570:
        return "Green/Yellow"
    elif wavelength_nm < 590:
        return "Orange"
    elif wavelength_nm < 620:
        return "Red/Orange"
    elif wavelength_nm < 750:
        return "Red"
    else:
        return "Infrared or beyond visible"

##############################
# SMILES Correction & Descriptor Calculation
##############################
def better_smiles_correction(smiles: str) -> tuple[str, bool]:
    """
    Attempt to correct the provided SMILES string.
    Returns a tuple: (corrected_smiles, was_corrected)
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol:
            return Chem.MolToSmiles(mol), False
    except Exception:
        pass

    corrected_smiles = smiles
    corrected = False
    if '/' in corrected_smiles or '\\' in corrected_smiles:
        corrected_smiles = re.sub(r'[\\/]', '', corrected_smiles)
        corrected = True
    ring_numbers = re.findall(r'\d', corrected_smiles)
    if len(ring_numbers) % 2 != 0:
        corrected_smiles = re.sub(r'\d', '', corrected_smiles, count=1)
        corrected = True
    open_brackets = corrected_smiles.count('[')
    close_brackets = corrected_smiles.count(']')
    if open_brackets > close_brackets:
        corrected_smiles = corrected_smiles.replace('[', '', open_brackets - close_brackets)
        corrected = True
    elif close_brackets > open_brackets:
        corrected_smiles = corrected_smiles.replace(']', '', close_brackets - open_brackets)
        corrected = True
    return corrected_smiles, corrected

def identify_chromophores(smiles: str) -> str:
    """Identify chromophore patterns within the SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 'Invalid SMILES'
    matches = [name for name, pattern in CHROMOPHORE_PATTERNS.items() 
               if pattern and mol.HasSubstructMatch(pattern)]
    return ', '.join(matches) if matches else 'Unknown'

def identify_auxochromes(smiles: str) -> str:
    """Identify auxochrome patterns within the SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 'Invalid SMILES'
    matches = [name for name, pattern in AUXOCHROME_PATTERNS.items() 
               if pattern and mol.HasSubstructMatch(pattern)]
    return ', '.join(matches) if matches else 'None'

def calc_num_double_bonds(mol: Chem.Mol) -> int:
    """Calculate the number of double bonds in the molecule."""
    return sum(1 for bond in mol.GetBonds() 
               if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)

def calc_num_rings(mol: Chem.Mol) -> int:
    """Calculate the number of rings in the molecule."""
    return mol.GetRingInfo().NumRings()

def get_conjugation_length(mol: Chem.Mol) -> int:
    """Determine the maximum conjugation length in the molecule."""
    if mol is None:
        return 0
    longest = 0
    def dfs(atom, visited, length):
        nonlocal longest
        longest = max(longest, length)
        for bond in atom.GetBonds():
            if bond.GetIsConjugated():
                neighbor = bond.GetOtherAtom(atom)
                if neighbor.GetIdx() not in visited:
                    visited.add(neighbor.GetIdx())
                    dfs(neighbor, visited, length + 1)
                    visited.remove(neighbor.GetIdx())
    for atom in mol.GetAtoms():
        dfs(atom, {atom.GetIdx()}, 0)
    return longest

def calculate_descriptors(smiles: str) -> dict:
    """Calculate a set of molecular descriptors for the provided SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {
            'MolWeight': None, 'LogP': None, 'TPSA': None,
            'NumRings': None, 'NumDoubleBonds': None, 'ConjugationLength': None
        }
    try:
        return {
            'MolWeight': Descriptors.MolWt(mol) if Descriptors else None,
            'LogP': Descriptors.MolLogP(mol) if Descriptors else None,
            'TPSA': rdMolDescriptors.CalcTPSA(mol) if rdMolDescriptors else None,
            'NumRings': calc_num_rings(mol),
            'NumDoubleBonds': calc_num_double_bonds(mol),
            'ConjugationLength': get_conjugation_length(mol)
        }
    except Exception:
        return {
            'MolWeight': None, 'LogP': None, 'TPSA': None,
            'NumRings': None, 'NumDoubleBonds': None, 'ConjugationLength': None
        }

def chromophore_to_color(chromophore: str) -> str:
    """Map chromophore type to a rough color description."""
    mapping = {
        'Azo': 'Red/Orange/Yellow',
        'Anthraquinone': 'Red/Blue/Violet',
        'Nitro': 'Yellow/Orange',
        'Quinone': 'Yellow/Orange/Brown',
        'Indigoid': 'Blue/Purple',
        'Cyanine': 'Green/Blue',
        'Xanthene': 'Yellow/Orange',
        'Thiazine': 'Blue/Green',
        'Coumarin': 'Blue/Green',
        'Porphyrin': 'Red/Purple',
        'Phthalocyanine': 'Green/Blue',
        'Carotenoid': 'Yellow/Orange',
        'Squaraine': 'Red/Purple',
        'Bromine': 'Dark Green/Purple',
        'Selenium': 'Deep Blue/Purple',
        'Pyridine': 'Varies (often Green/Blue/Yellow)',
        'Phosphine': 'Varies (Yellow/Green)',
        'Carbene': 'Varies (Red/Purple)',
        'Metal Complex': 'Varies (often Green/Blue)'
    }
    return mapping.get(chromophore, 'Unknown')

def estimate_color(chromophores: str, auxochromes: str, descriptors: dict, smiles: str) -> tuple[str, int]:
    """
    Estimate the color description and λmax based on identified chromophores,
    auxochromes, and calculated molecular descriptors.
    """
    if chromophores == 'Invalid SMILES':
        return ('Invalid SMILES', 0)
    if chromophores != 'Unknown':
        first_chromo = chromophores.split(', ')[0]
        approx_nm = estimate_lambda_max_scientific(smiles, first_chromo)
        base_color = chromophore_to_color(first_chromo)
    else:
        approx_nm = estimate_lambda_max_scientific(smiles)
        base_color = nm_to_color_category(approx_nm)
    
    if auxochromes != 'Invalid SMILES':
        auxo_list = [aux.strip() for aux in auxochromes.split(',')]
        if 'Hydroxyl' in auxo_list:
            base_color += ' (Shifted towards Red)'
        if 'Amine' in auxo_list:
            base_color += ' (Shifted towards Blue/Violet)'
        if 'Methoxy' in auxo_list:
            base_color += ' (Shifted towards Yellow)'
        if 'Thiol' in auxo_list:
            base_color += ' (Potential for Increased Color Intensity)'
        if 'Carboxyl' in auxo_list:
            base_color += ' (Potential for Increased Solubility)'
    return base_color, approx_nm

##############################
# Data Processing and Caching
##############################
@st.cache_data
def process_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the uploaded CSV file by validating and correcting SMILES, calculating
    descriptors, and estimating colors.
    """
    if 'SMILES_Corrected' not in df.columns:
        df['SMILES_Corrected'] = False
    if 'SMILES_Valid' not in df.columns:
        df['SMILES_Valid'] = False
    if 'Corrected_SMILES' not in df.columns:
        df['Corrected_SMILES'] = df['SMILES']
    
    # Process each SMILES entry
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            corrected_smiles, was_corrected = better_smiles_correction(smiles)
            df.at[idx, 'Corrected_SMILES'] = corrected_smiles
            if was_corrected:
                df.at[idx, 'SMILES_Corrected'] = True
                mol = Chem.MolFromSmiles(corrected_smiles)
                df.at[idx, 'SMILES_Valid'] = bool(mol)
            else:
                df.at[idx, 'SMILES_Valid'] = False
        else:
            df.at[idx, 'SMILES_Valid'] = True

    df['Chromophore'] = df['Corrected_SMILES'].apply(identify_chromophores)
    df['Auxochrome'] = df['Corrected_SMILES'].apply(identify_auxochromes)
    df['Descriptors'] = df['Corrected_SMILES'].apply(calculate_descriptors)
    
    # Expand descriptor dictionary into separate columns
    descriptor_df = df['Descriptors'].apply(pd.Series)
    for col in ['MolWeight', 'LogP', 'TPSA', 'NumRings', 'NumDoubleBonds', 'ConjugationLength']:
        if col not in descriptor_df.columns:
            descriptor_df[col] = None
    df = pd.concat([df, descriptor_df], axis=1)
    df.drop(columns=['Descriptors'], inplace=True)
    
    # Estimate color based on chromophore/auxochrome and descriptors
    color_results = df.apply(
        lambda x: estimate_color(
            x['Chromophore'],
            x['Auxochrome'],
            {
                'MolWeight': x['MolWeight'],
                'LogP': x['LogP'],
                'TPSA': x['TPSA'],
                'NumRings': x['NumRings'],
                'NumDoubleBonds': x['NumDoubleBonds'],
                'ConjugationLength': x['ConjugationLength']
            },
            x['Corrected_SMILES']
        ),
        axis=1
    )
    df['Estimated Color'] = color_results.apply(lambda tup: tup[0])
    df['ApproxLambda'] = color_results.apply(lambda tup: tup[1])
    return df

def convert_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert specified DataFrame columns to numeric."""
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

##############################
# Visualization Functions
##############################
def visualize_smiles_2d(smiles: str):
    """Generate a 2D image of the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol and Draw:
        try:
            return Draw.MolToImage(mol, size=(250, 250))
        except Exception as e:
            st.error(f"2D drawing failed: {e}")
            return None
    else:
        st.error("2D drawing functionality is not available.")
        return None

def visualize_smiles_3d(smiles: str) -> str:
    """Generate a 3D visualization HTML for the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Invalid SMILES"
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        return f"3D embedding failed: {e}"
    mb = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=350, height=300)
    view.addModel(mb, 'mol')
    view.setStyle({'stick': {}})
    view.zoomTo()
    return view._make_html()

##############################
# Streamlit UI Functions
##############################
def display_csv_workflow(visualize_option: str) -> None:
    """Handle CSV upload, processing, visualization, and download."""
    uploaded_file = st.file_uploader("Upload a CSV file with a 'SMILES' column...", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return

    if 'SMILES' not in df.columns:
        st.error("The CSV must contain a 'SMILES' column.")
        return

    with st.spinner("Processing your CSV..."):
        df = process_file(df)

    df = convert_numeric_columns(df, ['ApproxLambda', 'MolWeight', 'LogP'])
    st.subheader("Analysis Results")
    st.markdown("Below is the processed data with estimated colors and descriptors:")

    # Color coding for estimated color column
    def color_background(val: str) -> str:
        text_lower = val.lower()
        if "red" in text_lower or "orange" in text_lower:
            return "background-color: #ffebe6;"
        if "blue" in text_lower or "violet" in text_lower or "purple" in text_lower:
            return "background-color: #ebe6ff;"
        if "green" in text_lower:
            return "background-color: #e6ffeb;"
        if "yellow" in text_lower or "brown" in text_lower:
            return "background-color: #f9ffe6;"
        return "background-color: #ffffff;"

    styled_df = df.style.applymap(color_background, subset=['Estimated Color'])
    st.dataframe(styled_df, use_container_width=True)

    # Molecular weight histogram
    st.subheader("Molecular Weight Distribution")
    chart_mw = alt.Chart(df.dropna(subset=['MolWeight'])).mark_bar(color='#9b59b6').encode(
        alt.X('MolWeight:Q', bin=alt.Bin(maxbins=30), title='Molecular Weight'),
        y='count()'
    ).properties(width=600, height=400)
    st.altair_chart(chart_mw, use_container_width=True)

    # λmax histogram
    st.subheader("λmax Distribution")
    chart_lambda = alt.Chart(df.dropna(subset=['ApproxLambda']).query("ApproxLambda > 0")).mark_bar(color='#FF6F61').encode(
        alt.X('ApproxLambda:Q', bin=alt.Bin(maxbins=25), title='Approx λmax (nm)'),
        y='count()',
        tooltip=[alt.Tooltip('count()', title='Count')]
    ).properties(width=600, height=400)
    st.altair_chart(chart_lambda, use_container_width=True)

    # Scatter plot for MolWeight vs LogP colored by Chromophore
    st.subheader("Descriptor Correlation")
    scatter_chart = alt.Chart(df.dropna(subset=['MolWeight', 'LogP'])).mark_circle(size=60).encode(
        x=alt.X('MolWeight:Q', title='Molecular Weight'),
        y=alt.Y('LogP:Q', title='LogP'),
        color=alt.Color('Chromophore:N', title='Chromophore'),
        tooltip=['SMILES', 'Chromophore', 'Estimated Color', 'ApproxLambda']
    ).properties(width=600, height=400).interactive()
    st.altair_chart(scatter_chart, use_container_width=True)

    # Display molecular structure previews for each entry
    st.subheader("Molecular Previews")
    for idx, row in df.iterrows():
        st.markdown(f"**Index {idx}** | **Estimated Color:** {row['Estimated Color']}")
        if row['SMILES_Valid']:
            if visualize_option == "2D":
                img = visualize_smiles_2d(row['Corrected_SMILES'])
                if img:
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.image(Image.open(buf))
                else:
                    st.write("2D image rendering failed.")
            else:
                st.components.v1.html(visualize_smiles_3d(row['Corrected_SMILES']), height=300)
        else:
            st.write(f"Index {idx}: Invalid SMILES, no preview available.")

    # Provide CSV download of enhanced data
    st.subheader("Download Enhanced Data")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="output_dye_colors_enhanced.csv",
        mime="text/csv"
    )

def display_single_smiles(visualize_option: str) -> None:
    """Handle single SMILES input and display its analysis and visualization."""
    user_smiles = st.text_input("Enter a SMILES string:")
    if not user_smiles:
        return

    tmp_df = pd.DataFrame({"SMILES": [user_smiles]})
    with st.spinner("Analyzing your SMILES..."):
        result_df = process_file(tmp_df)
    result_df = convert_numeric_columns(result_df, ['ApproxLambda', 'MolWeight', 'LogP'])
    row = result_df.iloc[0]

    # Display detailed analysis in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Corrected SMILES:**", row['Corrected_SMILES'])
        st.write("**Valid SMILES?**", row['SMILES_Valid'])
        st.write("**Chromophores:**", row['Chromophore'])
        st.write("**Auxochromes:**", row['Auxochrome'])
    with col2:
        st.write("**Molecular Weight:**", row['MolWeight'])
        st.write("**LogP:**", row['LogP'])
        st.write("**TPSA:**", row['TPSA'])
        st.write("**Number of Rings:**", row['NumRings'])
        st.write("**Double Bonds:**", row['NumDoubleBonds'])
        st.write("**Conjugation Length:**", row['ConjugationLength'])
    st.write("**Approx λmax:**", row['ApproxLambda'])
    st.write("**Estimated Color:**", row['Estimated Color'])

    if row['SMILES_Valid']:
        st.subheader("Structure Preview")
        if visualize_option == "2D":
            img = visualize_smiles_2d(row['Corrected_SMILES'])
            if img:
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.image(Image.open(buf))
            else:
                st.write("2D image rendering failed.")
        else:
            st.components.v1.html(visualize_smiles_3d(row['Corrected_SMILES']), height=300)
    else:
        st.write("The provided SMILES is invalid, unable to generate a structure preview.")

##############################
# Main App Entry Point
##############################
def main() -> None:
    """Main function to run the Enhanced Chromophore & Auxochrome Analysis App."""
    st.title("Enhanced Chromophore & Auxochrome Analysis App")
    st.markdown(
        """
        *Welcome to the futuristic SMILES analyzer!*  
        **Features:**  
        - Intelligent SMILES correction  
        - Chromophore & Auxochrome identification  
        - Heuristic λmax estimation and color prediction  
        - Interactive 2D/3D molecular visualizations  
        - Comprehensive descriptor analysis and plots  
        
        Use the sidebar to select your preferred input method and visualization style.
        """
    )
    st.sidebar.header("Input Options")
    method = st.sidebar.radio("Method:", ["CSV Upload", "Single SMILES Input"])
    visualize_option = st.sidebar.radio("Visualization Type:", ["2D", "3D"])
    if method == "CSV Upload":
        display_csv_workflow(visualize_option)
    else:
        display_single_smiles(visualize_option)

if __name__ == "__main__":
    main()