import asyncio
import traceback

from dotenv import load_dotenv
from lightrag import QueryParam
from lightrag import LightRAG
from utils import initialize_rag



def test_with_sample_data(rag: LightRAG):
    """Test with sample data if LLM is available"""
    print("\n" + "="*50)
    print("Testing with Sample Data")
    print("="*50)
    
    try:
        # Insert documents
        rag.insert([
            """AI is transforming healthcare by enabling precise diagnostics through pattern recognition. The integration of artificial intelligence in medical diagnostics represents one of the most significant technological advances in modern medicine. Healthcare providers are increasingly relying on AI systems to analyze complex medical data, identify patterns that human physicians might miss, and provide diagnostic recommendations with unprecedented accuracy.

            These AI-powered diagnostic systems utilize deep learning algorithms trained on millions of medical cases, enabling them to recognize subtle patterns in medical imaging, laboratory results, and patient symptoms. The technology has proven particularly effective in radiology, where AI can detect early-stage cancers, identify fractures, and spot abnormalities in CT scans, MRIs, and X-rays with remarkable precision.

            Beyond imaging, AI diagnostic tools are being deployed across various medical specialties including cardiology, dermatology, ophthalmology, and pathology. In cardiology, AI algorithms can analyze ECGs to predict heart attacks hours before traditional methods. In dermatology, AI systems can examine skin lesions and moles to identify potential melanomas with accuracy rates comparable to experienced dermatologists.

            The implementation of AI diagnostics is not just improving accuracy but also addressing critical healthcare challenges such as physician shortages, especially in underserved areas. Telemedicine platforms enhanced with AI diagnostic capabilities are bringing expert-level medical analysis to remote locations where specialists are not readily available.

            However, the adoption of AI in diagnostics also raises important considerations around medical liability, data privacy, and the need for proper validation and regulation. Healthcare institutions must ensure that AI systems are properly integrated into clinical workflows while maintaining the human element of medical care that patients value.""",

            """Machine learning algorithms analyze medical images for early disease detection with 95% accuracy. The field of medical imaging has been revolutionized by advanced machine learning techniques that can process and interpret radiological images with superhuman precision. These sophisticated algorithms, particularly convolutional neural networks (CNNs) and transformer-based models, have been trained on vast datasets containing millions of annotated medical images.

            The 95% accuracy rate represents a significant milestone in medical AI, often surpassing the diagnostic accuracy of individual radiologists and matching the performance of expert consensus panels. This level of accuracy has been achieved across multiple imaging modalities including mammography for breast cancer screening, chest X-rays for pneumonia detection, retinal photographs for diabetic retinopathy, and brain MRIs for tumor identification.

            The early detection capabilities of these systems are particularly transformative for cancer care, where early intervention can dramatically improve patient outcomes. For instance, AI systems can identify lung nodules as small as 3mm in chest CT scans, detect breast cancer up to two years before it becomes clinically apparent, and spot signs of Alzheimer's disease in brain scans decades before symptoms manifest.

            The machine learning pipeline for medical image analysis involves several critical steps: data preprocessing and augmentation, feature extraction, model training with cross-validation, and rigorous testing on diverse patient populations. Advanced techniques such as federated learning are being employed to train models across multiple institutions while preserving patient privacy.

            Quality assurance and continuous monitoring are essential components of deployed systems. These algorithms undergo regular performance evaluations and are updated as new training data becomes available. Integration with picture archiving and communication systems (PACS) ensures seamless workflow integration in clinical settings.

            The economic impact is substantial, with AI-assisted imaging reducing interpretation time by up to 50% while improving diagnostic confidence. This efficiency gain allows radiologists to focus on complex cases requiring human expertise while ensuring that routine screenings are processed rapidly and accurately.""",

            """Natural language processing helps doctors extract insights from patient records efficiently. Electronic health records (EHRs) contain vast amounts of unstructured clinical data that has historically been difficult to analyze systematically. Natural language processing (NLP) technologies are transforming how healthcare providers extract meaningful insights from clinical notes, discharge summaries, pathology reports, and other textual medical documentation.

            Modern NLP systems employ sophisticated techniques including named entity recognition (NER), sentiment analysis, clinical concept extraction, and temporal reasoning to parse complex medical language. These systems can identify medications, dosages, adverse reactions, family history, social determinants of health, and treatment responses from free-text clinical notes with remarkable accuracy.

            Clinical decision support systems powered by NLP can alert physicians to potential drug interactions, suggest differential diagnoses based on symptom patterns, and identify patients at risk for specific conditions. For example, NLP algorithms can scan emergency department notes to identify patients showing early signs of sepsis, potentially saving lives through earlier intervention.

            The technology is particularly valuable for clinical research and population health management. Researchers can use NLP to identify patient cohorts for clinical trials, track treatment outcomes across large populations, and conduct retrospective studies that would be prohibitively expensive using manual chart review. Public health officials can monitor disease outbreaks by analyzing clinical notes for symptom patterns and geographic clusters.

            Advanced NLP applications include clinical summarization, where algorithms generate concise summaries of lengthy patient records for physician review, and clinical coding assistance, where systems suggest appropriate ICD-10 and CPT codes based on clinical documentation. This automation reduces administrative burden and improves coding accuracy.

            Privacy-preserving NLP techniques such as differential privacy and secure multi-party computation enable analysis of sensitive medical data while protecting patient confidentiality. De-identification algorithms can automatically remove or mask personal health information from clinical texts, facilitating research while maintaining HIPAA compliance.

            The integration of NLP with voice recognition technology is creating new possibilities for physician-computer interaction, enabling hands-free documentation and real-time clinical decision support during patient encounters.""",

            """Robotic surgery systems provide enhanced precision and minimal invasive procedures. The evolution of robotic surgery represents a paradigm shift in surgical practice, combining advanced robotics, computer vision, and haptic feedback to enable surgeons to perform complex procedures with unprecedented precision and control. These sophisticated systems, such as the da Vinci Surgical System and emerging platforms like the Intuitive ION for lung biopsy, are transforming surgical outcomes across multiple specialties.

            The enhanced precision of robotic systems stems from their ability to eliminate natural hand tremor, provide 10x magnification of the surgical field, and offer 360-degree instrument articulation that exceeds human wrist capabilities. Surgeons operate through a console that translates their hand movements into precise micro-movements of robotic instruments, allowing for intricate procedures in confined anatomical spaces.

            Minimally invasive robotic procedures typically require only small incisions, resulting in reduced blood loss, shorter hospital stays, faster recovery times, and minimal scarring compared to traditional open surgery. Patients undergoing robotic procedures often experience 50% less pain, 40% shorter recovery periods, and significantly lower risk of surgical complications.

            The technology has proven particularly transformative in urological procedures, gynecological surgery, cardiac surgery, and general surgery. In prostatectomy procedures, robotic surgery has dramatically improved nerve-sparing techniques, leading to better preservation of urinary and sexual function. In cardiac surgery, robotic systems enable surgeons to perform mitral valve repairs through small chest incisions rather than full sternotomy.

            Advanced features of modern robotic systems include augmented reality overlays that provide surgeons with real-time anatomical guidance, AI-powered surgical analytics that can predict optimal surgical approaches, and integrated imaging systems that combine pre-operative scans with live surgical video. Some systems now incorporate machine learning algorithms that can assist with tissue identification and provide real-time feedback on surgical technique.

            Training and certification programs for robotic surgery have become increasingly sophisticated, utilizing virtual reality simulators that allow surgeons to practice procedures in risk-free environments. These training platforms track performance metrics and provide objective assessments of surgical skills.

            The future of robotic surgery includes autonomous surgical tasks, where robots can perform specific procedures with minimal human intervention, and micro-robotics for cellular-level surgical interventions. Research is also advancing in soft robotics for more natural tissue interaction and swarm robotics for coordinated multi-robot surgical procedures.""",

            """AI-powered drug discovery accelerates the development of new treatments by 10x. The pharmaceutical industry is experiencing a revolutionary transformation through the application of artificial intelligence in drug discovery and development. Traditional drug discovery processes, which typically take 10-15 years and cost billions of dollars, are being dramatically accelerated through AI-driven approaches that can identify promising compounds, predict their properties, and optimize their therapeutic potential in a fraction of the time.

            Machine learning algorithms are being employed across every stage of the drug discovery pipeline, from target identification and validation to compound screening, lead optimization, and clinical trial design. AI systems can analyze vast databases of molecular structures, protein interactions, and genetic information to identify novel drug targets and predict which compounds are most likely to succeed in clinical development.

            Virtual screening powered by AI has revolutionized the process of identifying potential drug candidates. Instead of physically testing millions of compounds in laboratory assays, AI algorithms can computationally evaluate molecular libraries containing billions of virtual compounds, identifying the most promising candidates for synthesis and testing. This approach has reduced the time required for initial compound identification from years to weeks.

            Deep learning models are particularly effective at predicting drug-target interactions, metabolic pathways, toxicity profiles, and off-target effects. These predictions help researchers identify potential safety issues early in development and optimize compounds to maximize efficacy while minimizing adverse effects. AI models can predict ADMET properties (absorption, distribution, metabolism, excretion, and toxicity) with increasing accuracy, helping prioritize compounds with favorable drug-like characteristics.

            The COVID-19 pandemic demonstrated the power of AI-accelerated drug discovery, with several therapeutic candidates identified and advanced to clinical trials in record time. AI platforms analyzed existing drugs for repurposing opportunities, identified novel antiviral compounds, and predicted optimal combination therapies.

            Personalized medicine represents another frontier where AI is making significant contributions. Machine learning algorithms can analyze patient genetic profiles, biomarker data, and clinical histories to predict individual responses to specific treatments. This approach enables the development of precision therapies tailored to specific patient populations or even individual patients.

            AI is also transforming clinical trial design and execution. Algorithms can identify optimal patient populations, predict enrollment rates, design adaptive trial protocols, and monitor safety signals in real-time. Digital biomarkers derived from wearable devices and smartphone sensors provide continuous monitoring capabilities that can detect treatment effects more sensitively than traditional clinical endpoints.

            Collaborative AI platforms are emerging that allow pharmaceutical companies, academic institutions, and research organizations to share data and computational resources while maintaining proprietary information security. These platforms accelerate discovery by providing access to larger datasets and more sophisticated AI models than any single organization could develop independently."""
        ])
    
        # Query with different approaches
        queries = [
            "How is AI used in healthcare?",
            "What are the benefits of machine learning in medical imaging?",
            "Tell me about AI in surgery",
            "How does NLP help doctors?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            result = rag.query(query, param=QueryParam(mode="late-interaction"))
            print(f"Result: {result}")
    
        
    except Exception as e:
        print(f"❌ Sample data test setup failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    load_dotenv()
    rag = asyncio.run(initialize_rag("test", "ColbertVectorDBStorage"))
    test_with_sample_data(rag)