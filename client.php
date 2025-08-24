<?php

class NLLBTranslator {
    private $apiUrl;
    
    public function __construct($apiUrl = 'http://localhost:5000') {
        $this->apiUrl = rtrim($apiUrl, '/');
    }
    
    /**
     * Check if the translation service is healthy
     */
    public function healthCheck() {
        $url = $this->apiUrl . '/health';
        
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, 10);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        if ($httpCode === 200) {
            return json_decode($response, true);
        }
        
        return false;
    }
    
    /**
     * Get supported languages
     */
    public function getSupportedLanguages() {
        $url = $this->apiUrl . '/languages';
        
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, 10);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        if ($httpCode === 200) {
            $data = json_decode($response, true);
            return $data['supported_languages'] ?? [];
        }
        
        return [];
    }
    
    /**
     * Translate text
     */
    public function translate($text, $sourceLanguage, $targetLanguage) {
        $url = $this->apiUrl . '/translate';
        
        $data = [
            'text' => $text,
            'source_language' => $sourceLanguage,
            'target_language' => $targetLanguage
        ];
        
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, 30);
        curl_setopt($ch, CURLOPT_HTTPHEADER, [
            'Content-Type: application/json',
            'Content-Length: ' . strlen(json_encode($data))
        ]);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);
        
        if ($error) {
            throw new Exception("cURL Error: " . $error);
        }
        
        if ($httpCode !== 200) {
            $errorData = json_decode($response, true);
            $errorMessage = $errorData['error'] ?? 'Unknown error';
            throw new Exception("Translation API Error (HTTP $httpCode): $errorMessage");
        }
        
        return json_decode($response, true);
    }
    
    /**
     * Translate to Swahili
     */
    public function translateToSwahili($text, $sourceLanguage = 'english') {
        return $this->translate($text, $sourceLanguage, 'swahili');
    }
    
    /**
     * Translate to Kinyarwanda
     */
    public function translateToKinyarwanda($text, $sourceLanguage = 'english') {
        return $this->translate($text, $sourceLanguage, 'kinyarwanda');
    }
    
    /**
     * Translate from Swahili
     */
    public function translateFromSwahili($text, $targetLanguage = 'english') {
        return $this->translate($text, 'swahili', $targetLanguage);
    }
    
    /**
     * Translate from Kinyarwanda
     */
    public function translateFromKinyarwanda($text, $targetLanguage = 'english') {
        return $this->translate($text, 'kinyarwanda', $targetLanguage);
    }
}

// Example usage
try {
    // Initialize translator (replace with your Docker container URL)
    $translator = new NLLBTranslator('http://your-coolify-domain:5000');
    
    // Check if service is healthy
    $health = $translator->healthCheck();
    if ($health && $health['status'] === 'healthy') {
        echo "Translation service is healthy!\n";
    } else {
        throw new Exception("Translation service is not available");
    }
    
    // Get supported languages
    $languages = $translator->getSupportedLanguages();
    echo "Supported languages: " . implode(', ', $languages) . "\n";
    
    // Example translations
    $text = "Hello, how are you today?";
    
    // English to Swahili
    $swahiliResult = $translator->translateToSwahili($text);
    echo "English to Swahili: " . $swahiliResult['translated_text'] . "\n";
    
    // English to Kinyarwanda
    $kinyarwandaResult = $translator->translateToKinyarwanda($text);
    echo "English to Kinyarwanda: " . $kinyarwandaResult['translated_text'] . "\n";
    
    // Swahili to English
    $swahiliText = "Habari za leo?";
    $englishResult = $translator->translateFromSwahili($swahiliText);
    echo "Swahili to English: " . $englishResult['translated_text'] . "\n";
    
} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
}

?>
