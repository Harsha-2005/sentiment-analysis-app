import streamlit as st
from transformers import pipeline
import time
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("🎭 Advanced Sentiment Analysis")
st.markdown("""
This app uses Hugging Face's `transformers` library to analyze text sentiment.
The model classifies text as **POSITIVE** or **NEGATIVE** with a confidence score.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings & Configuration")
    
    # Model selection
    model_name = st.selectbox(
        "Choose a model:",
        [
            "distilbert-base-uncased-finetuned-sst-2-english",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ],
        index=0,
        help="Different models have different specialties and languages"
    )
    
    # Batch processing option
    batch_mode = st.checkbox("Enable batch processing", value=False)
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Minimum confidence threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.7,
        step=0.05,
        help="Filter results by confidence score"
    )
    
    # Max text length
    max_length = st.slider(
        "Maximum text length (characters)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50
    )
    
    st.divider()
    
    # File upload section in sidebar
    st.subheader("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Upload text file for batch analysis",
        type=['txt', 'csv'],
        help="Upload .txt (one text per line) or .csv with 'text' column"
    )
    
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
    Built with:
    - 🤗 Hugging Face Transformers
    - 🎈 Streamlit
    - 🐍 Python
    - 📊 Plotly for visualizations
    """)

# Initialize the classifier with caching (1 hour TTL)
@st.cache_resource(ttl=3600)
def load_classifier(_model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Load the sentiment analysis model with caching.
    Cache lasts for 1 hour to balance performance and updates.
    """
    try:
        with st.spinner(f"Loading {_model_name.split('/')[-1]} model..."):
            classifier = pipeline(
                "sentiment-analysis", 
                model=_model_name,
                truncation=True
            )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Trying to load default model...")
        try:
            classifier = pipeline("sentiment-analysis")
            return classifier
        except:
            return None

# Load the classifier
classifier = load_classifier(model_name)

# Function to process uploaded file
def process_uploaded_file(file):
    """Process uploaded file and extract texts"""
    try:
        if file.name.endswith('.txt'):
            texts = [line.strip() for line in file.read().decode("utf-8").split('\n') if line.strip()]
            return texts
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
            if 'text' in df.columns:
                texts = df['text'].dropna().astype(str).tolist()
                return texts
            else:
                st.error("CSV file must contain a 'text' column")
                return []
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return []

# Function to prepare download data
def prepare_download_data(results, texts):
    """Prepare data for download in multiple formats"""
    data = []
    for i, (text, result) in enumerate(zip(texts, results)):
        data.append({
            'id': i + 1,
            'text': text[:200] + "..." if len(text) > 200 else text,
            'full_text': text,
            'sentiment': result['label'],
            'confidence': result['score'],
            'confidence_percentage': f"{result['score']:.2%}",
            'timestamp': datetime.now().isoformat()
        })
    return data

# Main content layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Text Input")
    
    # Text input area
    default_text = "I've been waiting for a HuggingFace course my whole life."
    
    if batch_mode:
        text_input = st.text_area(
            "Enter text (one per line for batch processing):",
            height=200,
            placeholder="Type your text here...\nAdd multiple texts on separate lines.",
            value=default_text,
            key="batch_input",
            max_chars=max_length * 10  # Allow more characters for batch
        )
        texts = [line.strip() for line in text_input.split('\n') if line.strip()]
    else:
        text_input = st.text_area(
            "Enter text:",
            height=150,
            placeholder="Type your text here...",
            value=default_text,
            key="single_input",
            max_chars=max_length
        )
        texts = [text_input] if text_input.strip() else []
    
    # Process uploaded file if exists
    if uploaded_file is not None:
        uploaded_texts = process_uploaded_file(uploaded_file)
        if uploaded_texts:
            st.success(f"✅ Loaded {len(uploaded_texts)} texts from {uploaded_file.name}")
            if st.checkbox("Use uploaded file texts instead of manual input"):
                texts = uploaded_texts[:50]  # Limit to first 50 texts for performance

with col2:
    st.subheader("📊 Model Info")
    
    if classifier:
        model_info = model_name.split('/')[-1]
        st.success(f"✅ Model loaded: **{model_info}**")
        
        # Show model statistics
        if texts:
            total_chars = sum(len(t) for t in texts)
            st.metric("Texts to analyze", len(texts))
            st.metric("Total characters", total_chars)
    else:
        st.error("❌ Model not loaded")
    
    st.caption("💡 Tip: Upload files for bulk analysis")

# Analyze button
col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
with col_btn2:
    clear_button = st.button("🗑️ Clear Results", use_container_width=True)
with col_btn3:
    example_button = st.button("📋 Load Examples", use_container_width=True)

# Handle clear button
if clear_button:
    if 'results' in st.session_state:
        del st.session_state['results']
    st.rerun()

# Handle example button
if example_button:
    examples = [
        "I've been waiting for a HuggingFace course my whole life.",
        "This product is absolutely terrible and waste of money.",
        "The service was amazing and the staff were very helpful!",
        "I'm feeling neutral about this situation.",
        "The movie was disappointing despite the great cast.",
        "Absolutely love this app! It's incredibly useful.",
        "Worst experience ever, would not recommend to anyone.",
        "The quality exceeded my expectations, fantastic work!"
    ]
    example_text = "\n".join(examples)
    if batch_mode:
        st.session_state.batch_input = example_text
    else:
        st.session_state.single_input = examples[0]
    st.rerun()

# Process analysis
if analyze_button and texts:
    if classifier is None:
        st.error("Model failed to load. Please try again or select a different model.")
    else:
        with st.spinner(f"Analyzing {len(texts)} text(s)..."):
            try:
                # Truncate texts if too long
                truncated_texts = [text[:max_length] for text in texts]
                
                # Run sentiment analysis
                start_time = time.time()
                results = classifier(truncated_texts)
                processing_time = time.time() - start_time
                
                # Store in session state for download
                st.session_state.results = results
                st.session_state.texts = texts
                
                # Display results
                st.divider()
                st.subheader("📈 Analysis Results")
                
                # Summary statistics
                positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
                negative_count = len(results) - positive_count
                avg_confidence = sum(r['score'] for r in results) / len(results)
                
                # Summary metrics
                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                with col_sum1:
                    st.metric("Total Texts", len(results))
                with col_sum2:
                    st.metric("Positive", positive_count)
                with col_sum3:
                    st.metric("Negative", negative_count)
                with col_sum4:
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                st.caption(f"Processing time: {processing_time:.2f} seconds")
                
                # Detailed results
                with st.expander("📋 View Detailed Results", expanded=True):
                    for i, (text, result) in enumerate(zip(texts, results)):
                        with st.container():
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                # Truncate display text
                                display_text = text[:150] + "..." if len(text) > 150 else text
                                st.markdown(f"**Text {i+1}:** `{display_text}`")
                                
                                # Confidence indicator
                                confidence = result['score']
                                if confidence >= 0.9:
                                    confidence_label = "Very High"
                                elif confidence >= 0.7:
                                    confidence_label = "High"
                                elif confidence >= 0.5:
                                    confidence_label = "Medium"
                                else:
                                    confidence_label = "Low"
                                
                                st.caption(f"Confidence: **{confidence_label}** ({confidence:.2%})")
                            
                            with col_b:
                                label = result['label']
                                score = result['score']
                                
                                # Determine color and emoji
                                if label == "POSITIVE":
                                    color = "🟢"
                                    emoji = "😊"
                                    bg_color = "#d4edda"
                                else:
                                    color = "🔴"
                                    emoji = "😟"
                                    bg_color = "#f8d7da"
                                
                                # Display with colored background
                                st.markdown(f"""
                                <div style='background-color: {bg_color}; padding: 10px; border-radius: 5px; text-align: center;'>
                                    <h3>{emoji}</h3>
                                    <h4>{color} {label}</h4>
                                    <p>{score:.2%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Progress bar
                            st.progress(
                                score,
                                text=f"Confidence Score: {score:.2%}"
                            )
                            
                            # Filter indicator
                            if score < confidence_threshold:
                                st.warning(f"⚠️ Below confidence threshold ({confidence_threshold:.0%})")
                            
                            if i < len(texts) - 1:
                                st.divider()
                
                # Visualization section
                if len(results) > 1:
                    st.divider()
                    st.subheader("📊 Visualizations")
                    
                    # Create DataFrame for visualizations
                    df_results = pd.DataFrame({
                        'Text': [f"Text {i+1}" for i in range(len(texts))],
                        'Sentiment': [r['label'] for r in results],
                        'Confidence': [r['score'] for r in results],
                        'Text_Short': [t[:50] + "..." if len(t) > 50 else t for t in texts]
                    })
                    
                    # Tabbed visualizations
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Distribution", "Confidence Scores", "Sentiment Map"])
                    
                    with viz_tab1:
                        # Pie chart
                        fig_pie = px.pie(
                            df_results, 
                            names='Sentiment',
                            title="Sentiment Distribution",
                            color='Sentiment',
                            color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red'},
                            hole=0.3
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with viz_tab2:
                        # Bar chart of confidence scores
                        fig_bar = px.bar(
                            df_results.sort_values('Confidence', ascending=False),
                            x='Text',
                            y='Confidence',
                            color='Sentiment',
                            title="Confidence Scores by Text",
                            color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red'},
                            hover_data=['Text_Short']
                        )
                        fig_bar.update_layout(xaxis_title="Text Number", yaxis_title="Confidence Score")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with viz_tab3:
                        # Scatter plot
                        if len(results) > 2:
                            fig_scatter = px.scatter(
                                df_results,
                                x=range(len(results)),
                                y='Confidence',
                                color='Sentiment',
                                size='Confidence',
                                title="Sentiment Confidence Map",
                                color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red'},
                                hover_data=['Text_Short'],
                                labels={'x': 'Text Index', 'y': 'Confidence'}
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.info("Need at least 3 texts for sentiment map visualization")
                
                # Download section
                st.divider()
                st.subheader("💾 Download Results")
                
                if 'results' in st.session_state:
                    download_data = prepare_download_data(results, texts)
                    
                    # Convert to different formats
                    json_data = json.dumps(download_data, indent=2)
                    df_download = pd.DataFrame(download_data)
                    csv_data = df_download.to_csv(index=False)
                    
                    # Download buttons in columns
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col_dl3:
                        # Excel download requires additional package
                        try:
                            import io
                            excel_buffer = io.BytesIO()
                            df_download.to_excel(excel_buffer, index=False, engine='openpyxl')
                            excel_data = excel_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download Excel",
                                data=excel_data,
                                file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        except:
                            st.info("Install `openpyxl` for Excel download")
                
                # Filtered results based on confidence threshold
                filtered_results = [(t, r) for t, r in zip(texts, results) if r['score'] >= confidence_threshold]
                if len(filtered_results) < len(results):
                    st.info(f"📊 **Filtered View:** Showing {len(filtered_results)}/{len(results)} texts above {confidence_threshold:.0%} confidence threshold")
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Try reducing the number of texts or text length.")

elif analyze_button:
    st.warning("⚠️ Please enter some text to analyze!")

# Example section
with st.expander("💡 Quick Examples & Tips"):
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.markdown("**Try these examples:**")
        examples_short = [
            "This is the best day ever!",
            "I'm extremely disappointed with the service.",
            "The product works perfectly, highly recommended!",
            "Waste of time and money."
        ]
        
        for example in examples_short:
            if st.button(example, key=f"ex_{example[:10]}"):
                if batch_mode:
                    current = st.session_state.get('batch_input', '')
                    st.session_state.batch_input = current + "\n" + example if current else example
                else:
                    st.session_state.single_input = example
                st.rerun()
    
    with col_ex2:
        st.markdown("**Tips for better analysis:**")
        st.markdown("""
        1. **Clear text**: Remove unnecessary symbols
        2. **Context matters**: The model works best with complete sentences
        3. **Length**: Very short texts (<5 words) may have lower confidence
        4. **Language**: Use English for best results with default models
        5. **Batch size**: For large batches, consider splitting into multiple runs
        """)

# Footer
st.divider()
st.caption(f"""
Built with ❤️ using Streamlit & Hugging Face Transformers | 
Model: {model_name.split('/')[-1]} | 
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

# Add a refresh button in footer
if st.button("🔄 Refresh App"):
    st.cache_resource.clear()
    st.rerun()
