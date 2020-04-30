import json
import logging
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
import requests
from django.views.decorators.csrf import csrf_exempt
# Create your views here.


logger = logging.getLogger(__name__)

@csrf_exempt
def chatbot(request):
    

    if request.method == "GET":
        return render(request, "chat.html")
    

    try:
        body= json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    user_message = body.get("message","").strip()

    if not user_message:
        return JsonResponse({'error': 'Message cannot be empty'}, status=400)
    
    # Optional role (system prompt) from the client
    role_message = body.get("role", "").strip()
    
    api_key = getattr(settings, 'OPENROUTER_API_KEY', None)

    if not api_key:
        logger.error("OpenRouter API key is not configured.")
        return JsonResponse({'error': 'Server configuration error'}, status=500)
    
    # Use provided role or fall back to default from settings
    if not role_message:
        role_message = getattr(settings, 'DEFAULT_BOT_ROLE', '')

    try:
        # Build request payload so we can inspect it if debugging
        payload = {
            "model": "mistralai/mistral-7b-instruct",
            # include a system message (role) first so the model adopts the character
            "messages": [
                {"role": "system", "content": role_message},
                {"role": "user", "content": user_message}
            ],
        }

        resp = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=15,
        )

    except requests.RequestException as exc:
        logger.exception("Error communicating with OpenAI API: %s", exc)
        return JsonResponse({'error': 'Error communicating with AI service'}, status=502)

    if resp.status_code != 200:
        logger.error("OpenAI API error %s: %s", resp.status_code, resp.text)
        return JsonResponse({'error': 'AI service error'}, status=502)
    
    try:
        data = resp.json()
        # Safely access content from the model response
        ai_reply = ''
        try:
            ai_reply = data.get('choices', [])[0].get('message', {}).get('content', '')
        except Exception:
            # fallback if structure is different
            ai_reply = data.get('choices', [])[0].get('text', '') if data.get('choices') else ''

        # If the model returned an empty reply, provide a helpful fallback so UI isn't blank
        attempts = []

        if not (ai_reply and ai_reply.strip()):
            # Record first (empty) attempt
            attempts.append({
                'type': 'initial',
                'response': data,
            })

            # Attempt 1: retry with more generous generation params
            try:
                retry_payload = payload.copy()
                retry_payload.update({
                    'max_tokens': 200,
                    'temperature': 0.7,
                })
                r2 = requests.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json',
                    },
                    json=retry_payload,
                    timeout=15,
                )
                data2 = r2.json() if r2.status_code == 200 else {'error_status': r2.status_code, 'text': r2.text}
                attempts.append({'type': 'retry_more_tokens', 'request': retry_payload, 'response': data2})
                # Try to extract reply
                try:
                    ai_reply = data2.get('choices', [])[0].get('message', {}).get('content', '')
                except Exception:
                    ai_reply = data2.get('choices', [])[0].get('text', '') if data2.get('choices') else ''
            except Exception as exc:
                logger.exception('Retry attempt 1 failed: %s', exc)

            # Attempt 2: retry without system role (in case role silences the model)
            if not (ai_reply and ai_reply.strip()):
                try:
                    payload_no_role = {
                        'model': payload.get('model'),
                        'messages': [
                            {'role': 'user', 'content': user_message}
                        ],
                        'max_tokens': 200,
                        'temperature': 0.7,
                    }
                    r3 = requests.post(
                        'https://openrouter.ai/api/v1/chat/completions',
                        headers={
                            'Authorization': f'Bearer {api_key}',
                            'Content-Type': 'application/json',
                        },
                        json=payload_no_role,
                        timeout=15,
                    )
                    data3 = r3.json() if r3.status_code == 200 else {'error_status': r3.status_code, 'text': r3.text}
                    attempts.append({'type': 'retry_no_role', 'request': payload_no_role, 'response': data3})
                    try:
                        ai_reply = data3.get('choices', [])[0].get('message', {}).get('content', '')
                    except Exception:
                        ai_reply = data3.get('choices', [])[0].get('text', '') if data3.get('choices') else ''
                except Exception as exc:
                    logger.exception('Retry attempt 2 failed: %s', exc)

            # Still empty after retries -> fallback
            if not (ai_reply and ai_reply.strip()):
                ai_reply = 'No response from the AI. Please try rephrasing your question.'
                try:
                    logger.debug('Model response JSON (attempts): %s', attempts)
                except Exception:
                    logger.exception('Failed to log model response attempts')

                if getattr(settings, 'DEBUG', False):
                    return JsonResponse({'reply': ai_reply, 'debug_attempts': attempts, 'sent_role': role_message, 'request_payload': payload})
    
    except Exception as exc:
        logger.exception("Error processing OpenAI API response: %s", exc)
        return JsonResponse({'error': 'Error processing AI response'}, status=500)

    return JsonResponse({'reply': ai_reply})