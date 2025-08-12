#!/usr/bin/env python3
"""
AnchorReceipts CLI
Command-line interface for interacting with the AnchorReceipts smart contract
"""

import argparse
import json
import sys
import os
from typing import List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import AnchorClient


def cmd_anchor_root(args):
    """Anchor a Merkle root"""
    client = AnchorClient(
        rpc_url=args.rpc_url,
        contract_address=args.contract,
        private_key=args.private_key
    )
    
    # Parse tags
    tags = args.tags.split(',') if args.tags else []
    
    print(f"Anchoring root for dispute: {args.dispute_id}")
    print(f"  Merkle Root: {args.merkle_root}")
    print(f"  Model Hash: {args.model_hash}")
    print(f"  Round: {args.round}")
    print(f"  URI: {args.uri}")
    print(f"  Tags: {tags}")
    
    try:
        result = client.anchor_root(
            dispute_id=args.dispute_id,
            merkle_root=args.merkle_root,
            model_hash=args.model_hash,
            round=args.round,
            uri=args.uri,
            tags=tags
        )
        
        print("\nSuccess!")
        print(f"Transaction Hash: {result['tx_hash']}")
        print(f"Block Number: {result['block_number']}")
        print(f"Gas Used: {result['gas_used']}")
        
        if 'event' in result:
            print("\nEvent Data:")
            print(json.dumps(result['event'], indent=2))
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_anchor_receipt(args):
    """Anchor a receipt"""
    client = AnchorClient(
        rpc_url=args.rpc_url,
        contract_address=args.contract,
        private_key=args.private_key
    )
    
    print(f"Anchoring receipt for dispute: {args.dispute_id}")
    print(f"  Receipt Hash: {args.receipt_hash}")
    print(f"  Step Index: {args.step_index}")
    print(f"  URI: {args.uri}")
    
    try:
        result = client.anchor_receipt(
            dispute_id=args.dispute_id,
            receipt_hash=args.receipt_hash,
            step_index=args.step_index,
            uri=args.uri
        )
        
        print("\nSuccess!")
        print(f"Transaction Hash: {result['tx_hash']}")
        print(f"Block Number: {result['block_number']}")
        print(f"Gas Used: {result['gas_used']}")
        
        if 'event' in result:
            print("\nEvent Data:")
            print(json.dumps(result['event'], indent=2))
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_latest_root(args):
    """Query the latest root for a dispute"""
    client = AnchorClient(
        rpc_url=args.rpc_url,
        contract_address=args.contract
    )
    
    print(f"Querying latest root for dispute: {args.dispute_id}")
    
    try:
        result = client.latest_root(args.dispute_id)
        
        print("\nLatest Root:")
        print(f"  Merkle Root: {result['merkle_root']}")
        print(f"  Block Number: {result['block_number']}")
        
        # Get detailed info if requested
        if args.detailed:
            info = client.latest_root_info(args.dispute_id)
            print(f"  Timestamp: {info['timestamp']}")
            if info['timestamp'] > 0:
                dt = datetime.fromtimestamp(info['timestamp'])
                print(f"  Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_get_round(args):
    """Get root for a specific round"""
    client = AnchorClient(
        rpc_url=args.rpc_url,
        contract_address=args.contract
    )
    
    print(f"Querying round {args.round} for dispute: {args.dispute_id}")
    
    try:
        result = client.get_root_by_round(args.dispute_id, args.round)
        
        print(f"\nRound {args.round} Root:")
        print(f"  Merkle Root: {result['merkle_root']}")
        print(f"  Block Number: {result['block_number']}")
        print(f"  Timestamp: {result['timestamp']}")
        
        if result['timestamp'] > 0:
            dt = datetime.fromtimestamp(result['timestamp'])
            print(f"  Date: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_info(args):
    """Get contract information"""
    client = AnchorClient(
        rpc_url=args.rpc_url,
        contract_address=args.contract
    )
    
    print("Fetching contract information...")
    
    try:
        info = client.get_contract_info()
        
        print("\nContract Information:")
        print(f"  Address: {info['address']}")
        print(f"  Total Roots Anchored: {info['total_roots']}")
        print(f"  Total Receipts Anchored: {info['total_receipts']}")
        print(f"\nNetwork Information:")
        print(f"  Chain ID: {info['network']['chain_id']}")
        print(f"  Current Block: {info['network']['block_number']}")
        print(f"  Gas Price: {info['network']['gas_price']} wei")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_check(args):
    """Check if a dispute has a root"""
    client = AnchorClient(
        rpc_url=args.rpc_url,
        contract_address=args.contract
    )
    
    print(f"Checking dispute: {args.dispute_id}")
    
    try:
        has_root = client.has_root(args.dispute_id)
        
        if has_root:
            print(f"\n✓ Dispute has anchored roots")
            
            # Get latest round
            latest_round = client.get_latest_round(args.dispute_id)
            print(f"  Latest Round: {latest_round}")
            
            # Get latest root info
            info = client.latest_root_info(args.dispute_id)
            print(f"  Latest Root: {info['merkle_root']}")
            print(f"  Block Number: {info['block_number']}")
            
            if info['timestamp'] > 0:
                dt = datetime.fromtimestamp(info['timestamp'])
                print(f"  Last Updated: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"\n✗ No roots anchored for this dispute")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='AnchorReceipts Smart Contract CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Anchor a root
  %(prog)s anchor-root -d dispute1 -m 0x123... -h 0x456... -r 1 -u ipfs://Qm... -t PoDP,FL
  
  # Anchor a receipt
  %(prog)s anchor-receipt -d dispute1 -rh 0x789... -s 5 -u ipfs://Qm...
  
  # Query latest root
  %(prog)s latest-root -d dispute1
  
  # Get contract info
  %(prog)s info
        """
    )
    
    # Global arguments
    parser.add_argument('--rpc-url', '-r', 
                       default=os.getenv('RPC_URL', 'http://127.0.0.1:8545'),
                       help='Ethereum RPC endpoint URL')
    parser.add_argument('--contract', '-c',
                       default=os.getenv('ANCHOR_ADDRESS'),
                       help='Contract address')
    parser.add_argument('--private-key', '-k',
                       default=os.getenv('PRIVATE_KEY'),
                       help='Private key for transactions')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # anchor-root command
    anchor_root_parser = subparsers.add_parser('anchor-root', 
                                               help='Anchor a Merkle root')
    anchor_root_parser.add_argument('-d', '--dispute-id', required=True,
                                    help='Dispute identifier')
    anchor_root_parser.add_argument('-m', '--merkle-root', required=True,
                                    help='Merkle root to anchor')
    anchor_root_parser.add_argument('-h', '--model-hash', required=True,
                                    help='Model hash')
    anchor_root_parser.add_argument('-r', '--round', type=int, required=True,
                                    help='Round number')
    anchor_root_parser.add_argument('-u', '--uri', required=True,
                                    help='IPFS or other URI')
    anchor_root_parser.add_argument('-t', '--tags', default='',
                                    help='Comma-separated tags')
    anchor_root_parser.set_defaults(func=cmd_anchor_root)
    
    # anchor-receipt command
    anchor_receipt_parser = subparsers.add_parser('anchor-receipt',
                                                  help='Anchor a receipt')
    anchor_receipt_parser.add_argument('-d', '--dispute-id', required=True,
                                       help='Dispute identifier')
    anchor_receipt_parser.add_argument('-rh', '--receipt-hash', required=True,
                                       help='Receipt hash')
    anchor_receipt_parser.add_argument('-s', '--step-index', type=int, required=True,
                                       help='Step index')
    anchor_receipt_parser.add_argument('-u', '--uri', required=True,
                                       help='IPFS or other URI')
    anchor_receipt_parser.set_defaults(func=cmd_anchor_receipt)
    
    # latest-root command
    latest_root_parser = subparsers.add_parser('latest-root',
                                               help='Get latest root for a dispute')
    latest_root_parser.add_argument('-d', '--dispute-id', required=True,
                                    help='Dispute identifier')
    latest_root_parser.add_argument('--detailed', action='store_true',
                                    help='Show detailed information')
    latest_root_parser.set_defaults(func=cmd_latest_root)
    
    # get-round command
    get_round_parser = subparsers.add_parser('get-round',
                                             help='Get root for a specific round')
    get_round_parser.add_argument('-d', '--dispute-id', required=True,
                                  help='Dispute identifier')
    get_round_parser.add_argument('-r', '--round', type=int, required=True,
                                  help='Round number')
    get_round_parser.set_defaults(func=cmd_get_round)
    
    # info command
    info_parser = subparsers.add_parser('info',
                                        help='Get contract information')
    info_parser.set_defaults(func=cmd_info)
    
    # check command
    check_parser = subparsers.add_parser('check',
                                         help='Check if dispute has roots')
    check_parser.add_argument('-d', '--dispute-id', required=True,
                             help='Dispute identifier')
    check_parser.set_defaults(func=cmd_check)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Check contract address for commands that need it
    if not args.contract:
        print("Error: Contract address not provided. Use --contract or set ANCHOR_ADDRESS env var.", 
              file=sys.stderr)
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()