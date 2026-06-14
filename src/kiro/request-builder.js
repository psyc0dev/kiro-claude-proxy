/**
 * Request Builder for Kiro/AWS CodeWhisperer
 *
 * Builds request payloads and headers for the AWS CodeWhisperer API.
 * Converts Anthropic format to AWS CodeWhisperer format.
 */

import crypto from 'crypto';
import {
    KIRO_MODEL_MAPPING,
    KIRO_HEADERS,
    isThinkingModel
} from '../constants.js';

/**
 * Map an Anthropic model name to Kiro's internal model ID
 * @param {string} anthropicModel - The Anthropic-format model name
 * @returns {string} The Kiro/CodeWhisperer model ID
 */
export function mapModelToKiro(anthropicModel) {
    const lower = (anthropicModel || '').toLowerCase();
    
    // Check direct mappings first
    if (KIRO_MODEL_MAPPING[lower]) {
        return KIRO_MODEL_MAPPING[lower];
    }
    
    // Fuzzy matching for common patterns
    if (lower.includes('opus')) {
        if (lower.includes('4.8')) return 'claude-opus-4.8';
        if (lower.includes('4.7')) return 'claude-opus-4.7';
        if (lower.includes('4.6')) return 'claude-opus-4.6';
        if (lower.includes('4.5')) return 'claude-opus-4.5';
        // Default opus to the newest version
        return 'claude-opus-4.8';
    }
    if (lower.includes('sonnet')) {
        if (lower.includes('4.8')) return 'claude-sonnet-4.8';
        if (lower.includes('4.7')) return 'claude-sonnet-4.7';
        if (lower.includes('4.5')) return 'claude-sonnet-4.5';
        if (lower.includes('4')) return 'claude-sonnet-4';
        // Default sonnet to the newest version
        return 'claude-sonnet-4.8';
    }
    if (lower.includes('haiku')) {
        if (lower.includes('4.7')) return 'claude-haiku-4.7';
        return 'claude-haiku-4.5';
    }
    
    // Default to the newest Claude model for unknown models
    return 'claude-opus-4.8';
}

/**
 * Convert Anthropic tool definitions to CodeWhisperer toolSpecification format.
 * @param {Array} tools - Anthropic-format tool definitions
 * @returns {Array|undefined} CodeWhisperer tool specs, or undefined if none
 */
export function convertToolsToKiro(tools) {
    if (!Array.isArray(tools) || tools.length === 0) {
        return undefined;
    }
    return tools.map(tool => ({
        toolSpecification: {
            name: tool.name,
            description: tool.description || '',
            inputSchema: {
                json: tool.input_schema || { type: 'object', properties: {} }
            }
        }
    }));
}

/**
 * Normalize the content of an Anthropic tool_result block into CodeWhisperer's
 * toolResult content format ([{ text }] or [{ json }]).
 * @param {*} content - The tool_result content (string, array, or object)
 * @returns {Array<Object>} CodeWhisperer tool result content blocks
 */
function normalizeToolResultContent(content) {
    if (typeof content === 'string') {
        return [{ text: content }];
    }
    if (Array.isArray(content)) {
        return content.map(item => {
            if (typeof item === 'string') return { text: item };
            if (item && item.type === 'text') return { text: item.text };
            if (item && item.type === 'json') return { json: item.json ?? item };
            return { json: item };
        });
    }
    if (content && typeof content === 'object') {
        return [{ json: content }];
    }
    return [{ text: String(content ?? '') }];
}

/**
 * Process an Anthropic message's content into structured CodeWhisperer parts.
 * Tool uses and tool results are preserved as structured data rather than being
 * flattened into text, so the model can use native tool calling.
 * @param {string|Array} content - The Anthropic message content
 * @returns {{ text: string, toolUses: Array, toolResults: Array }}
 */
function processMessageContent(content) {
    const result = { text: '', toolUses: [], toolResults: [] };

    if (typeof content === 'string') {
        result.text = content;
        return result;
    }
    if (!Array.isArray(content)) {
        return result;
    }

    const textParts = [];
    for (const block of content) {
        switch (block.type) {
            case 'text':
                textParts.push(block.text || '');
                break;
            case 'thinking':
                textParts.push(`<thinking>${block.thinking}</thinking>`);
                break;
            case 'tool_use':
                result.toolUses.push({
                    toolUseId: block.id,
                    name: block.name,
                    input: block.input || {}
                });
                break;
            case 'tool_result':
                result.toolResults.push({
                    toolUseId: block.tool_use_id,
                    content: normalizeToolResultContent(block.content),
                    status: block.is_error ? 'error' : 'success'
                });
                break;
            case 'image':
                // CodeWhisperer text endpoint can't accept images; note as placeholder
                textParts.push('[Image attached]');
                break;
        }
    }
    result.text = textParts.join('\n');
    return result;
}

/**
 * Flatten an Anthropic system prompt (string or array of blocks) to text.
 * @param {string|Array} system - The system prompt
 * @returns {string} Flattened system text
 */
function flattenSystem(system) {
    if (!system) return '';
    if (typeof system === 'string') return system;
    if (Array.isArray(system)) {
        return system.map(s => (typeof s === 'string' ? s : s.text || '')).join('\n');
    }
    return String(system);
}

/**
 * Convert Anthropic message format to structured CodeWhisperer messages.
 * @param {Object} anthropicRequest - The Anthropic-format request
 * @returns {Object} Structured conversation data
 */
export function convertAnthropicToKiro(anthropicRequest) {
    const messages = anthropicRequest.messages || [];
    const processed = messages.map(msg => ({
        role: msg.role === 'assistant' ? 'assistant' : 'user',
        ...processMessageContent(msg.content)
    }));

    return {
        systemText: flattenSystem(anthropicRequest.system),
        messages: processed,
        tools: convertToolsToKiro(anthropicRequest.tools),
        maxTokens: anthropicRequest.max_tokens || 8192,
        temperature: anthropicRequest.temperature,
        topP: anthropicRequest.top_p
    };
}

/**
 * Build a CodeWhisperer history entry from a processed message.
 * @param {Object} msg - Processed message ({ role, text, toolUses, toolResults })
 * @returns {Object} CodeWhisperer history entry
 */
function buildHistoryEntry(msg) {
    if (msg.role === 'assistant') {
        const entry = { content: msg.text || '' };
        if (msg.toolUses.length > 0) {
            entry.toolUses = msg.toolUses;
        }
        return { assistantResponseMessage: entry };
    }

    const entry = { content: msg.text || '' };
    if (msg.toolResults.length > 0) {
        entry.userInputMessageContext = { toolResults: msg.toolResults };
    }
    return { userInputMessage: entry };
}

/**
 * Build the CodeWhisperer chat request payload.
 * Forwards tool definitions and tool results in CodeWhisperer's native format
 * so the model performs real tool calling instead of emitting tool tags as text.
 * @param {Object} anthropicRequest - The Anthropic-format request
 * @returns {Object} The CodeWhisperer API request payload
 */
export function buildKiroRequest(anthropicRequest) {
    const model = mapModelToKiro(anthropicRequest.model);
    const { systemText, messages, tools } = convertAnthropicToKiro(anthropicRequest);

    // The final message is the current turn; everything before it is history.
    const current = messages[messages.length - 1] || { role: 'user', text: '', toolUses: [], toolResults: [] };
    const previous = messages.slice(0, -1);

    // CodeWhisperer has no system role; prepend the system prompt to the
    // current user message so it still steers the response.
    let currentContent = current.text || '';
    if (systemText) {
        currentContent = currentContent
            ? `${systemText}\n\n${currentContent}`
            : systemText;
    }

    const userInputMessageContext = {
        editorState: { cursorState: null }
    };
    if (tools) {
        userInputMessageContext.tools = tools;
    }
    if (current.toolResults.length > 0) {
        userInputMessageContext.toolResults = current.toolResults;
    }

    return {
        conversationState: {
            conversationId: crypto.randomUUID(),
            chatTriggerType: 'MANUAL',
            customizationArn: null,
            currentMessage: {
                userInputMessage: {
                    content: currentContent,
                    userInputMessageContext,
                    modelId: model,
                    origin: 'AI_EDITOR'
                }
            },
            history: previous.map(buildHistoryEntry)
        },
        profileArn: null,
        source: 'AI_EDITOR',
        modelId: model,
        origin: 'AI_EDITOR'
    };
}

/**
 * Build headers for CodeWhisperer API requests
 * @param {string} token - AWS access token
 * @param {string} region - AWS region
 * @param {boolean} streaming - Whether this is a streaming request
 * @returns {Object} Headers object
 */
export function buildKiroHeaders(token, region = 'us-east-1', streaming = false) {
    const headers = {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
        'Accept': streaming ? 'application/vnd.amazon.eventstream' : 'application/json',
        'X-Amz-Region': region,
        ...KIRO_HEADERS
    };
    
    return headers;
}

/**
 * Build a simple chat completion request for testing
 * @param {string} prompt - The user prompt
 * @param {string} model - Model ID
 * @returns {Object} Simple request payload
 */
export function buildSimpleKiroRequest(prompt, model = 'auto') {
    return {
        conversationState: {
            conversationId: crypto.randomUUID(),
            chatTriggerType: 'MANUAL',
            currentMessage: {
                userInputMessage: {
                    content: prompt,
                    modelId: model,
                    origin: 'AI_EDITOR'
                }
            }
        },
        source: 'AI_EDITOR',
        modelId: model,
        origin: 'AI_EDITOR'
    };
}

export default {
    mapModelToKiro,
    convertAnthropicToKiro,
    convertToolsToKiro,
    buildKiroRequest,
    buildKiroHeaders,
    buildSimpleKiroRequest
};
