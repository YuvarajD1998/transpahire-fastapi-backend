from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional, Tuple
import jwt
from app.config import settings
from app.models.enums import SubscriptionTier

security = HTTPBearer()

class JWTHandler:
    """Handle both access and refresh tokens from NestJS service"""
    
    def __init__(self):
        self.access_token_secret = settings.jwt_secret
        self.refresh_token_secret = settings.jwt_refresh_secret
        self.algorithm = "HS256"
    
    def decode_token(self, token: str) -> Tuple[Dict[str, Any], str]:
        """
        Decode JWT token and determine if it's access or refresh token
        Returns: (payload, token_type)
        """
        try:
            # First, decode without verification to check token type
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            
            # Check if it's a refresh token
            is_refresh_token = unverified_payload.get('type') == 'refresh'
            
            # Use appropriate secret based on token type
            secret = self.refresh_token_secret if is_refresh_token else self.access_token_secret
            token_type = 'refresh' if is_refresh_token else 'access'
            
            # Decode with verification but disable strict subject validation
            payload = jwt.decode(
                token,
                secret,
                algorithms=[self.algorithm],
                options={
                    "verify_exp": True,
                    "verify_sub": False,  # Disable subject validation for integer IDs
                    "verify_signature": True
                }
            )
            
            return payload, token_type
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidSignatureError:
            # Try the other secret as fallback
            try:
                other_secret = self.access_token_secret if is_refresh_token else self.refresh_token_secret
                payload = jwt.decode(
                    token,
                    other_secret,
                    algorithms=[self.algorithm],
                    options={
                        "verify_exp": True,
                        "verify_sub": False,  # Disable subject validation
                        "verify_signature": True
                    }
                )
                token_type = 'access' if is_refresh_token else 'refresh'
                return payload, token_type
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token signature: {str(e)}"
                )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}"
            )

# Global JWT handler instance
jwt_handler = JWTHandler()

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Get current user from JWT token (handles both access and refresh tokens).
    """
    if not creds or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    
    token = creds.credentials
    
    try:
        # Decode the JWT token (handles both types)
        payload, token_type = jwt_handler.decode_token(token)
        
        # Ensure sub (user ID) is an integer
        user_id = payload.get('sub')
        if isinstance(user_id, str):
            try:
                user_id = int(user_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid user ID in token"
                )
        
        # For refresh tokens, we need to handle differently
        if token_type == 'refresh':
            # Refresh tokens have limited payload
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token payload"
                )
            
            # For refresh tokens, fetch user data from database
            user_data = await get_user_from_database(user_id)
            
            return {
                "id": user_id,
                "email": user_data.get("email", ""),
                "role": user_data.get("role", "CANDIDATE"),
                "tenant_id": None,
                "verified": user_data.get("verified", True),
                "status": user_data.get("status", "ACTIVE"),
                "subscription_tier": user_data.get("subscription_tier", "FREE"),
                "token_type": token_type,
                "iat": payload.get("iat"),
                "exp": payload.get("exp"),
            }
        
        else:  # Access token
            # Ensure this is a candidate (since this API is candidate-only)
            if payload.get("role") != "CANDIDATE":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="This endpoint is only available for candidates"
                )
            
            # For access tokens, get subscription info from database
            user_data = await get_user_from_database(user_id)
            
            return {
                "id": user_id,
                "email": payload.get("email"),
                "role": payload.get("role"),
                "tenant_id": payload.get("tenantId"),
                "verified": True,
                "status": "ACTIVE",
                "subscription_tier": user_data.get("subscription_tier", "FREE"),
                "token_type": token_type,
                "iat": payload.get("iat"),
                "exp": payload.get("exp"),
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token processing failed: {str(e)}"
        )

async def get_user_from_database(user_id: int) -> Dict[str, Any]:
    """
    Fetch user data from database.
    TODO: Implement actual database query
    """
    # Mock data for now
    return {
        "email": "candidate@example.com",
        "role": "CANDIDATE",
        "verified": True,
        "status": "ACTIVE",
        "subscription_tier": "PREMIUM"
    }

def require_subscription(min_tier: SubscriptionTier):
    """Require minimum subscription tier for candidates."""
    tier_order = {
        SubscriptionTier.FREE: 0, 
        SubscriptionTier.BASIC: 1, 
        SubscriptionTier.PREMIUM: 2
    }
    
    async def guard(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if user.get("role") != "CANDIDATE":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is only available for candidates"
            )
        
        user_tier = user.get("subscription_tier", "FREE")
        try:
            user_tier_enum = SubscriptionTier(user_tier)
        except ValueError:
            user_tier_enum = SubscriptionTier.FREE
            
        if tier_order[user_tier_enum] < tier_order[min_tier]:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"{min_tier.value} subscription required"
            )
        
        return user
    
    return guard
