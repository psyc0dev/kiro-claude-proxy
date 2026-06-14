/**
 * Streaming Handler for Kiro/AWS CodeWhisperer
 *
 * Handles streaming message requests using AWS CodeWhisperer API.
 * Yields Anthropic-format SSE events as they arrive.
 */

import crypto from 'crypto';
import {
    KIRO_ENDPOINTS,
    KIRO_API_PATHS,
    KIRO_DEFAULT_REGION,
    MAX_RETRIES
} from '../constants.js';
import { getKiroAuthData } from '../auth/kiro-token-extractor.js';
import { buildKiroRequest, buildKiroHeaders, mapModelToKiro } from './request-builder.js';
import { parseEventStreamAsync } from './aws-event-stream.js';
import { logger } from '../utils/logger.js';
import { sleep } from '../utils/helpers.js';

/**
 * Send a streaming request to Kiro/CodeWhisperer
 * Yields Anthropic-format SSE events in real-time
 *
 * @param {Object} anthropicRequest - The Anthropic-format request
 * @param {string} anthropicRequest.model - Model name to use
 * @param {Array} anthropicRequest.messages - Array of message objects
 * @param {number} [anthropicRequest.max_tokens] - Maximum tokens to generate
 * @yields {Object} Anthropic-format SSE events
 * @throws {Error} If request fails or no token available
 */
export async function* sendKiroMessageStream(anthropicRequest) {
    const model = anthropicRequest.model;
    const kiroModel = mapModelToKiro(model);
    
    logger.debug(`[Kiro] Starting stream for model: ${model} -> ${kiroModel}`);
    
    // Get auth data
    const authData = await getKiroAuthData();
    const token = authData.accessToken;
    const region = authData.region || KIRO_DEFAULT_REGION;
    
    if (!token) {
        throw new Error('No Kiro authentication token available. Please log in to Kiro CLI first.');
    }
    
    // Build the request payload
    const payload = buildKiroRequest(anthropicRequest);
    
    // Add model to header and request streaming response
    const headers = {
        ...buildKiroHeaders(token, region, true),
        'x-amzn-access-model': kiroModel,
        'Accept': 'application/vnd.amazon.eventstream'
    };
    
    // Get endpoint for this region
    const endpoint = KIRO_ENDPOINTS[region] || KIRO_ENDPOINTS[KIRO_DEFAULT_REGION];
    const url = `${endpoint}${KIRO_API_PATHS.GENERATE_ASSISTANT}`;
    
    logger.debug(`[Kiro] Stream URL: ${url}`);
    
    // Retry loop
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers,
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                logger.warn(`[Kiro] Stream error ${response.status}: ${errorText}`);
                
                if (response.status === 401) {
                    throw new Error('Kiro authentication expired. Please log in again.');
                }
                
                if (response.status === 429) {
                    const retryAfter = response.headers.get('retry-after');
                    const waitMs = retryAfter ? parseInt(retryAfter) * 1000 : 5000;
                    logger.warn(`[Kiro] Rate limited, waiting ${waitMs}ms...`);
                    await sleep(waitMs);
                    continue;
                }
                
                if (response.status >= 500) {
                    const waitMs = Math.pow(2, attempt) * 1000;
                    logger.warn(`[Kiro] Server error, retrying in ${waitMs}ms...`);
                    await sleep(waitMs);
                    continue;
                }
                
                throw new Error(`Kiro API error ${response.status}: ${errorText}`);
            }
            
            // Stream the response
            yield* streamKiroResponse(response, model);
            return; // Success, exit retry loop
            
        } catch (error) {
            if (error.message.includes('authentication') || 
                error.message.includes('expired')) {
                throw error;
            }
            
            if (attempt === MAX_RETRIES - 1) {
                throw error;
            }
            
            logger.warn(`[Kiro] Stream attempt ${attempt + 1} failed: ${error.message}`);
            await sleep(Math.pow(2, attempt) * 1000);
        }
    }
    
    throw new Error('Max retries exceeded');
}

/**
 * Stream and parse Kiro event stream response using AWS binary format.
 * Tracks text and tool-use content blocks, accumulating partial tool input
 * across events so tool calls are emitted as proper Anthropic tool_use blocks.
 * @param {Response} response - The fetch response
 * @param {string} requestModel - The original model requested
 * @yields {Object} Anthropic-format SSE events
 */
async function* streamKiroResponse(response, requestModel) {
    const messageId = `msg_${crypto.randomUUID().replace(/-/g, '').substring(0, 24)}`;
    let hasStarted = false;
    let nextIndex = 0;
    let outputTokens = 0;
    let inputTokens = 0;
    let stopReason = 'end_turn';

    // Current non-tool block: 'thinking' | 'text' | null
    let currentType = null;
    let currentIndex = -1;

    // Tool block state: toolUseId -> content block index
    const toolBlocks = new Map();

    function startMessage() {
        return {
            type: 'message_start',
            message: {
                id: messageId,
                type: 'message',
                role: 'assistant',
                model: requestModel,
                content: [],
                stop_reason: null,
                stop_sequence: null,
                usage: { input_tokens: 0, output_tokens: 0 }
            }
        };
    }

    try {
        for await (const event of parseEventStreamAsync(response.body)) {
            if (logger.isDebugEnabled) {
                logger.debug(`[Kiro] Raw event: ${JSON.stringify(event)}`);
            }
            if (!hasStarted) {
                hasStarted = true;
                yield startMessage();
            }

            // --- Tool use events ------------------------------------------
            // CodeWhisperer emits these flat: { name, toolUseId, input?, stop? }.
            // Also accept legacy wrapped forms (toolUseEvent / toolUse).
            const toolUse = event.toolUseEvent || event.toolUse ||
                (event.toolUseId !== undefined ? event : null);
            if (toolUse) {
                stopReason = 'tool_use';

                // Close any open text/thinking block before the tool block
                if (currentType !== null) {
                    yield { type: 'content_block_stop', index: currentIndex };
                    currentType = null;
                    currentIndex = -1;
                }

                const toolId = toolUse.toolUseId;

                // Open a new tool_use block the first time we see this id
                if (toolId !== undefined && !toolBlocks.has(toolId)) {
                    const newIdx = nextIndex++;
                    toolBlocks.set(toolId, newIdx);
                    yield {
                        type: 'content_block_start',
                        index: newIdx,
                        content_block: {
                            type: 'tool_use',
                            id: toolId,
                            name: toolUse.name || 'tool',
                            input: {}
                        }
                    };
                }

                const idx = toolBlocks.get(toolId);

                // Forward partial JSON input verbatim (it arrives in string chunks)
                if (idx !== undefined &&
                    toolUse.input !== undefined && toolUse.input !== null && toolUse.input !== '') {
                    const partial = typeof toolUse.input === 'string'
                        ? toolUse.input
                        : JSON.stringify(toolUse.input);
                    yield {
                        type: 'content_block_delta',
                        index: idx,
                        delta: { type: 'input_json_delta', partial_json: partial }
                    };
                }

                // Close the tool block when the model signals completion
                if (toolUse.stop && idx !== undefined) {
                    yield { type: 'content_block_stop', index: idx };
                    toolBlocks.delete(toolId);
                }
                continue;
            }

            // --- Token usage metadata -------------------------------------
            const usage = event.metadataEvent?.tokenUsage || event.tokenUsage;
            if (usage) {
                inputTokens = usage.inputTokens || inputTokens;
                outputTokens = usage.outputTokens || outputTokens;
                continue;
            }
            // Credit/usage metering and context stats - log/skip, not token counts
            if (event.usage !== undefined && event.unit !== undefined) {
                logger.debug(`[Kiro] Usage: ${event.usage} ${event.unitPlural || event.unit}`);
                continue;
            }
            if (event.contextUsagePercentage !== undefined) {
                continue;
            }

            // --- Extended thinking deltas (event.text) --------------------
            if (typeof event.text === 'string' && event.text.length > 0) {
                if (currentType !== 'thinking') {
                    if (currentType !== null) {
                        yield { type: 'content_block_stop', index: currentIndex };
                    }
                    currentIndex = nextIndex++;
                    currentType = 'thinking';
                    yield {
                        type: 'content_block_start',
                        index: currentIndex,
                        content_block: { type: 'thinking', thinking: '' }
                    };
                }
                yield {
                    type: 'content_block_delta',
                    index: currentIndex,
                    delta: { type: 'thinking_delta', thinking: event.text }
                };
                continue;
            }

            // --- Thinking signature ---------------------------------------
            if (typeof event.signature === 'string' && currentType === 'thinking') {
                yield {
                    type: 'content_block_delta',
                    index: currentIndex,
                    delta: { type: 'signature_delta', signature: event.signature }
                };
                continue;
            }

            // --- Visible text content -------------------------------------
            const text = extractEventText(event);
            if (text) {
                if (currentType !== 'text') {
                    if (currentType !== null) {
                        yield { type: 'content_block_stop', index: currentIndex };
                    }
                    currentIndex = nextIndex++;
                    currentType = 'text';
                    yield {
                        type: 'content_block_start',
                        index: currentIndex,
                        content_block: { type: 'text', text: '' }
                    };
                }
                yield {
                    type: 'content_block_delta',
                    index: currentIndex,
                    delta: { type: 'text_delta', text }
                };
            }
        }

        // Ensure message_start was emitted even for empty responses
        if (!hasStarted) {
            yield startMessage();
        }

        // Close any still-open text/thinking block
        if (currentType !== null) {
            yield { type: 'content_block_stop', index: currentIndex };
        }
        // Close any tool blocks that never received an explicit stop
        for (const idx of toolBlocks.values()) {
            yield { type: 'content_block_stop', index: idx };
        }

        // Emit final message_delta with stop reason and usage
        yield {
            type: 'message_delta',
            delta: { stop_reason: stopReason, stop_sequence: null },
            usage: { output_tokens: outputTokens }
        };

        yield { type: 'message_stop' };

    } catch (error) {
        logger.error(`[Kiro] Streaming error: ${error.message}`);
        throw error;
    }
}

/**
 * Extract text content from a Kiro/CodeWhisperer stream event.
 * @param {Object} event - Parsed event data
 * @returns {string} Extracted text, or empty string
 */
function extractEventText(event) {
    if (typeof event.content === 'string') {
        return event.content;
    }
    if (event.assistantResponseEvent?.content) {
        return event.assistantResponseEvent.content;
    }
    if (event.codeEvent?.content) {
        return event.codeEvent.content;
    }
    return '';
}

export default {
    sendKiroMessageStream
};
