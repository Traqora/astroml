import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

interface FAQItem {
  id: number;
  category: string;
  question: string;
  answer: string;
  order: number;
  is_published: boolean;
  created_at: string;
  updated_at: string;
}

interface FAQResponse {
  data: FAQItem[];
  categories: string[];
  total: number;
}

interface FeedbackStats {
  faq_id: number;
  helpful: number;
  not_helpful: number;
  total: number;
  helpful_percentage: number;
}

const FAQ: React.FC = () => {
  const { t } = useTranslation();
  const [faqs, setFaqs] = useState<FAQItem[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<Set<number>>(new Set());

  useEffect(() => {
    fetchFAQs();
  }, [selectedCategory, searchQuery]);

  const fetchFAQs = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (selectedCategory) params.append('category', selectedCategory);
      if (searchQuery) params.append('search', searchQuery);

      const response = await fetch(`/api/v1/faq?${params.toString()}`);
      const data: FAQResponse = await response.json();
      setFaqs(data.data);
      setCategories(data.categories);
    } catch (error) {
      console.error('Failed to fetch FAQs:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (faqId: number, isHelpful: boolean) => {
    if (feedbackSubmitted.has(faqId)) return;

    try {
      await fetch(`/api/v1/faq/${faqId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_helpful: isHelpful }),
      });

      setFeedbackSubmitted(new Set([...feedbackSubmitted, faqId]));
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    }
  };

  const toggleExpand = (id: number) => {
    setExpandedId(expandedId === id ? null : id);
  };

  const handleSuggest = async () => {
    const question = prompt('Enter your question:');
    if (!question) return;

    const suggestedAnswer = prompt('Suggested answer (optional):');
    const category = prompt('Category (optional):');

    try {
      await fetch('/api/v1/faq/suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          suggested_answer: suggestedAnswer || null,
          category: category || null,
        }),
      });
      alert('Thank you for your suggestion!');
    } catch (error) {
      console.error('Failed to submit suggestion:', error);
      alert('Failed to submit suggestion');
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Frequently Asked Questions</h1>
        <p className="text-gray-600">Find answers to common questions about AstroML</p>
      </div>

      {/* Search and Filter */}
      <div className="mb-6 space-y-4">
        <input
          type="text"
          placeholder="Search FAQs..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />

        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setSelectedCategory('')}
            className={`px-4 py-2 rounded-lg ${
              selectedCategory === ''
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            All
          </button>
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-lg capitalize ${
                selectedCategory === category
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* FAQ List */}
      {loading ? (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      ) : faqs.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No FAQs found. Try a different search or category.
        </div>
      ) : (
        <div className="space-y-4">
          {faqs.map((faq) => (
            <div
              key={faq.id}
              className="border border-gray-200 rounded-lg overflow-hidden"
            >
              <button
                onClick={() => toggleExpand(faq.id)}
                className="w-full px-6 py-4 text-left bg-white hover:bg-gray-50 flex justify-between items-center"
              >
                <div className="flex-1">
                  <span className="inline-block px-2 py-1 text-xs font-semibold bg-blue-100 text-blue-800 rounded mr-3 capitalize">
                    {faq.category}
                  </span>
                  <span className="font-medium">{faq.question}</span>
                </div>
                <svg
                  className={`w-5 h-5 text-gray-500 transition-transform ${
                    expandedId === faq.id ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>

              {expandedId === faq.id && (
                <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
                  <div className="prose prose-sm max-w-none mb-4">
                    {faq.answer}
                  </div>

                  {!feedbackSubmitted.has(faq.id) && (
                    <div className="flex items-center gap-4 pt-4 border-t border-gray-200">
                      <span className="text-sm text-gray-600">Was this helpful?</span>
                      <button
                        onClick={() => handleFeedback(faq.id, true)}
                        className="flex items-center gap-1 px-3 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
                      >
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                        Yes
                      </button>
                      <button
                        onClick={() => handleFeedback(faq.id, false)}
                        className="flex items-center gap-1 px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200"
                      >
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path
                            fillRule="evenodd"
                            d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                            clipRule="evenodd"
                          />
                        </svg>
                        No
                      </button>
                    </div>
                  )}

                  {feedbackSubmitted.has(faq.id) && (
                    <div className="pt-4 border-t border-gray-200">
                      <span className="text-sm text-green-600">Thank you for your feedback!</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Suggest FAQ Button */}
      <div className="mt-8 text-center">
        <button
          onClick={handleSuggest}
          className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          Suggest a New FAQ
        </button>
      </div>
    </div>
  );
};

export default FAQ;
