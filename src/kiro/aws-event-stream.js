/**
 * AWS Event Stream Parser for Kiro/CodeWhisperer
 * 
 * Parses the AWS binary event stream format used by CodeWhisperer APIs.
 * Format: [total_length:4][headers_length:4][prelude_crc:4][headers:headers_length][payload][message_crc:4]
 */

/**
 * Parse a single AWS event stream message from a buffer
 * @param {ArrayBuffer} buffer - The buffer containing the event stream data
 * @param {number} offset - The offset to start reading from
 * @returns {Object|null} Parsed event with data and next offset, or null if incomplete
 */
function parseEventMessage(buffer, offset) {
    if (offset + 12 > buffer.byteLength) {
        return null; // Not enough data for prelude
    }
    
    const view = new DataView(buffer, offset);
    const totalLength = view.getUint32(0);
    const headersLength = view.getUint32(4);
    // const preludeCrc = view.getUint32(8); // CRC check skipped for simplicity
    
    if (offset + totalLength > buffer.byteLength) {
        return null; // Incomplete message
    }
    
    const payloadOffset = offset + 12 + headersLength;
    const payloadLength = totalLength - headersLength - 16; // 12 prelude + 4 message CRC
    
    if (payloadLength <= 0) {
        return { data: null, nextOffset: offset + totalLength };
    }
    
    const bytes = new Uint8Array(buffer, payloadOffset, payloadLength);
    const payload = new TextDecoder().decode(bytes);
    
    let data = null;
    try {
        data = JSON.parse(payload);
    } catch (e) {
        // Not JSON, return raw string
        data = { raw: payload };
    }
    
    return {
        data,
        nextOffset: offset + totalLength
    };
}

/**
 * Parse all events from an AWS event stream buffer
 * @param {ArrayBuffer} buffer - The buffer containing event stream data
 * @returns {Array<Object>} Array of parsed events
 */
export function parseEventStream(buffer) {
    const events = [];
    let offset = 0;
    
    while (offset < buffer.byteLength) {
        const result = parseEventMessage(buffer, offset);
        if (!result) break;
        
        if (result.data !== null) {
            events.push(result.data);
        }
        offset = result.nextOffset;
    }
    
    return events;
}

/**
 * Parse AWS event stream from a ReadableStream (for streaming responses)
 * @param {ReadableStream} stream - The readable stream from fetch response
 * @yields {Object} Parsed events as they arrive
 */
export async function* parseEventStreamAsync(stream) {
    const reader = stream.getReader();
    let buffer = new Uint8Array(0);
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            // Append new data to buffer
            const newBuffer = new Uint8Array(buffer.length + value.length);
            newBuffer.set(buffer);
            newBuffer.set(value, buffer.length);
            buffer = newBuffer;
            
            // Parse complete messages from buffer
            let offset = 0;
            while (offset < buffer.length) {
                if (offset + 12 > buffer.length) break;
                
                const view = new DataView(buffer.buffer, buffer.byteOffset + offset);
                const totalLength = view.getUint32(0);
                
                if (offset + totalLength > buffer.length) break;
                
                const headersLength = view.getUint32(4);
                const payloadOffset = offset + 12 + headersLength;
                const payloadLength = totalLength - headersLength - 16;
                
                if (payloadLength > 0) {
                    const payload = new TextDecoder().decode(
                        buffer.slice(payloadOffset, payloadOffset + payloadLength)
                    );
                    
                    try {
                        yield JSON.parse(payload);
                    } catch (e) {
                        yield { raw: payload };
                    }
                }
                
                offset += totalLength;
            }
            
            // Keep unprocessed data in buffer
            if (offset > 0) {
                buffer = buffer.slice(offset);
            }
        }
    } finally {
        reader.releaseLock();
    }
}

/**
 * Extract content from parsed events
 * @param {Array<Object>} events - Array of parsed events
 * @returns {Object} Extracted content with text, usage, etc.
 */
export function extractContentFromEvents(events) {
    let fullContent = '';
    let usage = { input_tokens: 0, output_tokens: 0 };

    // Accumulate tool use input across events, keyed by tool use id.
    const toolOrder = [];
    const toolMap = new Map(); // toolUseId -> { id, name, inputParts: [] }

    for (const event of events) {
        // Text content
        if (typeof event.content === 'string') {
            fullContent += event.content;
        } else if (event.assistantResponseEvent?.content) {
            fullContent += event.assistantResponseEvent.content;
        } else if (event.codeEvent?.content) {
            fullContent += event.codeEvent.content;
        }

        // Token usage
        const tokenUsage = event.metadataEvent?.tokenUsage || event.tokenUsage;
        if (tokenUsage) {
            usage.input_tokens = tokenUsage.inputTokens || usage.input_tokens;
            usage.output_tokens = tokenUsage.outputTokens || usage.output_tokens;
        }

        // Tool use events (input may arrive in partial chunks). CodeWhisperer
        // emits these flat: { name, toolUseId, input?, stop? }. Also accept
        // legacy wrapped forms (toolUse / toolUseEvent).
        const toolUse = event.toolUse || event.toolUseEvent ||
            (event.toolUseId !== undefined ? event : null);
        if (toolUse) {
            const id = toolUse.toolUseId || `tool_${toolOrder.length}`;
            if (!toolMap.has(id)) {
                toolMap.set(id, { id, name: toolUse.name, inputParts: [] });
                toolOrder.push(id);
            }
            const entry = toolMap.get(id);
            if (toolUse.name && !entry.name) {
                entry.name = toolUse.name;
            }
            if (toolUse.input !== undefined && toolUse.input !== null) {
                entry.inputParts.push(toolUse.input);
            }
        }
    }

    // Finalize tool uses: join partial input and parse JSON when possible.
    const toolUses = toolOrder.map(id => {
        const entry = toolMap.get(id);
        let input = {};
        if (entry.inputParts.length > 0) {
            // If chunks are objects, take the last/merged object; if strings,
            // concatenate and parse as JSON.
            if (entry.inputParts.every(p => typeof p === 'object')) {
                input = Object.assign({}, ...entry.inputParts);
            } else {
                const joined = entry.inputParts.map(p =>
                    typeof p === 'string' ? p : JSON.stringify(p)
                ).join('');
                try {
                    input = joined ? JSON.parse(joined) : {};
                } catch {
                    input = { _raw: joined };
                }
            }
        }
        return { id: entry.id, name: entry.name, input };
    });

    return {
        content: fullContent,
        usage,
        toolUses
    };
}

export default {
    parseEventStream,
    parseEventStreamAsync,
    extractContentFromEvents
};
