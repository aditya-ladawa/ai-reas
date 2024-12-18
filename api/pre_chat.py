examples = [
    (
        "what is neural_networks.pdf talking about",
        {
            "query": "neural networks",
            "filter": 'eq("pdf_name", "neural_networks.pdf")',
        },
    ),
    (
        "Can you tell me something about the file cancer_research_study.pdf?",
        {
            "query": "cancer research",
            "filter": 'eq("pdf_name", "cancer_research_study.pdf")',
        },
    ),
    (
        "I need to know what ethics_in_ai.pdf says on ethical concerns.",
        {
            "query": "ethical concerns",
            "filter": 'eq("pdf_name", "ethics_in_ai.pdf")',
        },
    ),
    (
        "Find me anything related to quantum_computing_paper.pdf",
        {
            "query": "quantum computing",
            "filter": 'eq("pdf_name", "quantum_computing_paper.pdf")',
        },
    ),
    (
        "Are there any references to Einstein's theories in physics_papers.pdf?",
        {
            "query": "Einstein's theories",
            "filter": 'eq("pdf_name", "physics_papers.pdf")',
        },
    ),
    (
        "What's in the climate_change_analysis.pdf about global warming?",
        {
            "query": "global warming",
            "filter": 'eq("pdf_name", "climate_change_analysis.pdf")',
        },
    ),
    (
        "Show me papers discussing blockchain from blockchain_articles.pdf",
        {
            "query": "blockchain technology",
            "filter": 'eq("pdf_name", "blockchain_articles.pdf")',
        },
    ),
    (
        "Can you retrieve sections of deep_learning_basics.pdf?",
        {
            "query": "deep learning",
            "filter": 'eq("pdf_name", "deep_learning_basics.pdf")',
        },
    ),
    (
        "Who authored machine_learning_review.pdf and neural_network_overview.pdf?",
        {
            "query": "authors",
            "filter": 'or(eq("pdf_name", "machine_learning_review.pdf"), eq("pdf_name", "neural_network_overview.pdf"))',
        },
    ),
    (
        "Give me the details from data_analysis_guide.pdf",
        {
            "query": "data analysis",
            "filter": 'eq("pdf_name", "data_analysis_guide.pdf")',
        },
    ),
]